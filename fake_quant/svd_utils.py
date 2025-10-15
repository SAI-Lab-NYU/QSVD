import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import hadamard_utils
import rotation_utils
import model_utils
import quant_utils
import utils
import tqdm
import act_aware_utils
import os
import data_utils
import fisher_info_utils
import grad_info_utils
import logging

import os
import torch.distributed as dist
import datetime

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

# Add global variables to track the sum of ranks
total_rank_sum = 0
total_linear_count = 0

class SVDLinear(nn.Module):
    def __init__(self, U, S, V, bias=None, sigma_fuse="UV", had_K=None, K=-1, had_mode='hadamard') -> None:
        super().__init__()
        self.ALinear = nn.Linear(U.size(1), U.size(0), bias=bias is not None)

        if bias is not None:
            self.ALinear.bias.data = bias
        self.BLinear = nn.Linear(V.size(0), V.size(1), bias=False)
        self.truncation_rank = S.size(0)
        if sigma_fuse == "UV":
            self.ALinear.weight.data = U.mul(S.sqrt()).contiguous()
            self.BLinear.weight.data = V.t().mul(S.sqrt().view(-1, 1)).contiguous()
        elif sigma_fuse == "U":
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
        elif sigma_fuse == "V":
            self.ALinear.weight.data = U.contiguous()
            self.BLinear.weight.data = V.t().mul(S.view(-1, 1)).contiguous()
        elif sigma_fuse == "adaptive":
            eps = 1e-6
            scale = U.abs().max(dim=0).values/ (V.abs().max(dim=0).values + eps) # [c, r] -> [r]
            self.ALinear.weight.data = U.mul(S**(1/(scale+1))).contiguous()
            self.BLinear.weight.data = V.t().mul(S**(scale/(scale+1)).view(-1, 1)).contiguous()
        elif sigma_fuse == 'profile':
            self.ALinear.weight.data = U.mul(S).contiguous()
            self.BLinear.weight.data = V.t().contiguous()
            self.S = S
            self.U = U
            self.V = V
        elif float(sigma_fuse) <=1.0 and float(sigma_fuse)>=0:
            self.ALinear.weight.data = U.mul(S**(1-float(sigma_fuse))).contiguous()
            self.BLinear.weight.data = V.t().mul((S**float(sigma_fuse)).view(-1, 1)).contiguous()
        else:
            raise RuntimeError(f"Error: unsupported sigma mode {sigma_fuse}")
        self.had_K = had_K
        self.K = K
        self.had_mode = had_mode

        # Add for collecting latent distribution attributes
        self.collect_latent = False
        self.alinear_hook_handle = None
        self.latent_stats = None

    def apply_had_rank(self):
        had_K = self.had_K
        K = self.K
        had_mode = self.had_mode

        if K >0 and had_mode in ['rh', 'random']:
            W = self.ALinear # input
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(W_, had_K).to(device="cpu", dtype=dtype)
            W = self.BLinear # output
            dtype = W.weight.data.dtype
            W_ = W.weight.data.to(device=utils.get_dev(), dtype=torch.float64)
            W.weight.data = torch.matmul(had_K.T, W_).to(device="cpu", dtype=dtype)
            if W.bias is not None:
                b = W.bias.data.to(device=utils.get_dev(), dtype=torch.float64)
                W.bias.data = torch.matmul(had_K.T, b).to(device="cpu", dtype=dtype)
        elif K>0 and had_mode == 'hadamard':
            hadamard_utils.apply_exact_had_to_linear(self.ALinear, had_dim=-1, output=False)
            hadamard_utils.apply_exact_had_to_linear(self.BLinear, had_dim=-1, output=True)
        # del self.had_k

    def start_collecting_latent(self):
        """Start collecting the input distribution of ALinear"""
        self.collect_latent = True
        
        # Initialize statistics
        self.latent_stats = torch.zeros(self.ALinear.in_features, device=utils.get_dev())
        
        # Register forward hook
        def hook(module, input, output):
            if not self.collect_latent:
                return
            
            # Use abs_max method
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            self.latent_stats = torch.where(
                abs_max > self.latent_stats,
                abs_max,
                self.latent_stats,
            )
        
        self.alinear_hook_handle = self.ALinear.register_forward_hook(hook)
        
    def stop_collecting_latent(self):
        """Stop collecting the input distribution of ALinear"""
        self.collect_latent = False
        
        # Remove hook
        if hasattr(self, 'alinear_hook_handle') and self.alinear_hook_handle is not None:
            self.alinear_hook_handle.remove()
            self.alinear_hook_handle = None
    
    def get_latent_stats(self):
        """Get statistics of ALinear input"""
        if self.latent_stats is not None:
            return self.latent_stats.cpu()
        return None

    def apply_latent_smooth(self, alpha=0.5, eps=1e-6):
        if self.latent_stats is None or torch.all(self.latent_stats == 0):
            logging.info("No available latent space statistics, cannot smooth.")
            return False
            
        # Get compute device and original device
        compute_device = utils.get_dev()
        original_device = self.ALinear.weight.device
        original_dtype = self.ALinear.weight.dtype
        
        # Move statistics and weights to compute device
        latent_stats = self.latent_stats.to(compute_device)
        a_weight = self.ALinear.weight.data.to(compute_device)
        b_weight = self.BLinear.weight.data.to(compute_device)
        
        # Compute column-wise statistics of ALinear weights (corresponding to each latent dimension)
        weight_stats = a_weight.abs().amax(dim=0)  # max absolute value per column [rank]?

        # Compute scaling factors to ensure similar importance for each latent dimension
        # Use the reciprocal of the max absolute value as the scaling factor
        scale_factors = (latent_stats**alpha)/(weight_stats**(1-alpha))
        scale_factors = torch.clamp(scale_factors, min=eps)
        
        # Apply scaling factors to ALinear and BLinear weights
        # The input dimension of ALinear corresponds to the latent space, so scale by column
        a_weight_scaled = a_weight * scale_factors.view(1, -1) # [c, rank]
        
        # The output dimension of BLinear corresponds to the latent space, so scale by row
        # Also need to take the reciprocal of the scaling factor to keep the overall transformation unchanged
        inv_scale_factors = 1.0 / scale_factors 
        b_weight_scaled = b_weight * inv_scale_factors.view(-1, 1) # [rank, c]
        
        # Move results back to original device and dtype
        self.ALinear.weight.data = a_weight_scaled.to(device=original_device, dtype=original_dtype)
        self.BLinear.weight.data = b_weight_scaled.to(device=original_device, dtype=original_dtype)
        
        logging.info(f"Successfully smoothed latent space weights, scale factor range: {scale_factors.min().item():.4f} - {scale_factors.max().item():.4f}")
        return True

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1.,
        sigma_fuse="UV",
        rank_align=1.,
        had_rank=False,
        had_mode='hadamard',
        seed=0,
    ):
        global total_rank_sum, total_linear_count
        
        # Ensure param_ratio is a Python float, not a tensor
        if isinstance(param_ratio, torch.Tensor):
            param_ratio = float(param_ratio.item())
        param_ratio = min(param_ratio, 2)
        
        n_params = linear.weight.numel()
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(np.ceil(rank / rank_align) * rank_align)
        
        # Update total rank sum and linear layer count
        total_rank_sum += rank
        total_linear_count += 1

        # logging.info("rank", rank)
        if had_rank:
            utils.set_seed(seed)
            if had_mode == 'hadamard':
                K = 1
                had_K = None
            elif had_mode == 'rh':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                K = rank
            elif had_mode == 'random':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                K = rank
        # Correctly move weights to CUDA device
        w = linear.weight.data.float().to(utils.get_dev())
        if act_aware:
            scaling_diag_matrix = torch.ones(linear.in_features, device=utils.get_dev())  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # logging.info("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix.to(utils.get_dev())**alpha
                scaling_diag_matrix += 1e-6  # avoid zero division
                scaling_matrix_inv = None
                w = w * scaling_diag_matrix.view(1, -1)
            elif hasattr(linear, "scaling_diag_matrixS"):
                scaling_diag_matrix = linear.scaling_diag_matrixS.to(utils.get_dev())
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception as e:
                    logging.info("Warning: scaling_diag_matrix is not full rank!")
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                w = w @ scaling_diag_matrix.float()
        Us = []
        Ss = []
        Vs = []
        try:
            print(rank)
            # SVD decomposition
            U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False)

            # Low rank approximation to the target rank
            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]
            V = Vt.T
        except Exception as e:
            logging.info(f"SVD failed for {linear}: {e}")
            return nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
        if act_aware:
            if scaling_matrix_inv is None:
                V = V / scaling_diag_matrix.view(-1, 1)
            else:
                V =  scaling_matrix_inv.T.float() @ V
        Us = [U]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            bias = linear.bias.data
        else:
            bias = None

        # nan or inf check
        for S in Ss:
            if (S != S).any():
                logging.info("nan in S")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
        for U in Us:
            if (U != U).any():
                logging.info("nan in U")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )
        for V in Vs:
            if (V != V).any():
                logging.info("nan in V")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device)
                )

        assert len(Us) == len(Ss) == len(Vs) == 1
        
        if had_rank:
            new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse, had_K, K, had_mode)
        else:
            new_linear = SVDLinear(Us[0], Ss[0], Vs[0], bias, sigma_fuse)
        new_linear.to(linear.weight.dtype)
        return new_linear.cpu()


    @staticmethod
    def from_linearkv(
        linear: nn.Linear,
        linear1: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1.,
        sigma_fuse="UV",
        rank_align=1.,
        had_rank=False,
        had_mode='hadamard',
        seed=0,
    ):
        global total_rank_sum, total_linear_count
        
        # Ensure param_ratio is a Python float, not a tensor
        if isinstance(param_ratio, torch.Tensor):
            param_ratio = float(param_ratio.item())
        param_ratio = min(param_ratio, 2)
        
        # if param_ratio >= 1:
        #     return linear
        n_params = linear.weight.numel()
        
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear.in_features + linear.out_features)
        # rank align
        rank = int(np.ceil(rank / rank_align) * rank_align)
        
        # Update total rank sum and linear layer count (here calculated as 2 linear layers share a rank)
        total_rank_sum += rank
        total_linear_count += 1

        # logging.info("rank", rank)
        if had_rank:
            utils.set_seed(seed)
            if had_mode == 'hadamard':
                K = 1
                had_K = None
            elif had_mode == 'rh':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                K = rank
            elif had_mode == 'random':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                K = rank
                
        # Correctly move weights to CUDA device
        device = utils.get_dev()
        w = torch.cat([linear.weight.data.float(), linear1.weight.data.float()], dim=0).to(device)
        if act_aware:
            scaling_diag_matrix = torch.ones(linear.in_features, device=utils.get_dev())  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # logging.info("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear.scaling_diag_matrix.to(utils.get_dev())**alpha
                scaling_diag_matrix += 1e-6  # avoid zero division
                scaling_matrix_inv = None
                w = w * scaling_diag_matrix.view(1, -1)
            elif hasattr(linear, "scaling_diag_matrixS"):
                scaling_diag_matrix = linear.scaling_diag_matrixS.to(utils.get_dev())
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception as e:
                    logging.info("Warning: scaling_diag_matrix is not full rank!")
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                w = w @ scaling_diag_matrix.float()
        Us = []
        Ss = []
        Vs = []
        try:
            logging.info(f"Rank: {rank}")
            # SVD decomposition
            U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False)

            # Low rank approximation to the target rank
            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]
            V = Vt.T
        except Exception as e:
            logging.info(f"Fuse KV SVD failed for {linear}: {e}")
            logging.info(f"Matrix information: shape={w.size()}, whether contains NaN: {torch.isnan(w).any()}, whether contains Inf: {torch.isinf(w).any()}")
            logging.info(f"Matrix statistics: min={w.min().item()}, max={w.max().item()}, mean={w.mean().item()}, std={w.std().item()}")
            return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device)
                )
        if act_aware:
            if scaling_matrix_inv is None:
                V = V / scaling_diag_matrix.view(-1, 1)
            else:
                V =  scaling_matrix_inv.T.float() @ V
        U = U.view(2, -1, rank)
        Us = [U[0], U[1]]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            biask = linear.bias.data
            biasv = linear1.bias.data
        else:
            biask = None
            biasv = None

        # nan or inf check
        for S in Ss:
            if (S != S).any():
                logging.info("nan in S")
                # For from_linearkv method return two linear layers
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device)
                )
        for U in Us:
            if (U != U).any():
                logging.info("nan in U")
                # For from_linearkv method return two linear layers
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device)
                )
        for V in Vs:
            if (V != V).any():
                logging.info("nan in V")
                # For from_linearkv method return two linear layers
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device)
                )

        assert len(Us)/2 == len(Ss) == len(Vs) == 1
        
        if had_rank:
            new_linearK = SVDLinear(Us[0], Ss[0], Vs[0], biask, sigma_fuse, had_K, K, had_mode)
            new_linearV = SVDLinear(Us[1], Ss[0], Vs[0], biasv, sigma_fuse, had_K, K, had_mode)
        else:
            new_linearK = SVDLinear(Us[0], Ss[0], Vs[0], biask, sigma_fuse)
            new_linearV = SVDLinear(Us[1], Ss[0], Vs[0], biasv, sigma_fuse)
        new_linearK.to(linear.weight.dtype)
        new_linearV.to(linear.weight.dtype)
        return new_linearK.cpu(), new_linearV.cpu()
    
    @staticmethod
    def from_linearqkv(
        linear: nn.Linear,
        linear1: nn.Linear,
        linear2: nn.Linear,
        param_ratio: float,
        act_aware=False,
        ic_split=1,
        oc_split=1,
        alpha=1.,
        sigma_fuse="UV",
        rank_align=1.,
        had_rank=False,
        had_mode='hadamard',
        seed=0,
    ):
        global total_rank_sum, total_linear_count
        
        # Ensure param_ratio is a Python float, not a tensor
        if isinstance(param_ratio, torch.Tensor):
            param_ratio = float(param_ratio.item())
        param_ratio = min(param_ratio, 2)
        
        n_params = linear1.weight.numel()
        
        compressed_params = int(n_params * param_ratio)
        assert ic_split == 1 or oc_split == 1
        rank = compressed_params // (linear1.in_features + linear1.out_features)
        # rank align
        rank = int(np.ceil(rank / rank_align) * rank_align)
        
        # Update total rank sum and linear layer count (here calculated as 2 linear layers share a rank)
        total_rank_sum += rank
        total_linear_count += 1

        # logging.info("rank", rank)
        if had_rank:
            utils.set_seed(seed)
            if had_mode == 'hadamard':
                K = 1
                had_K = None
            elif had_mode == 'rh':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                K = rank
            elif had_mode == 'random':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                K = rank
                
        # Correctly move weights to CUDA device
        device = utils.get_dev()
        w = torch.cat([linear.weight.data.float(), linear1.weight.data.float(), linear2.weight.data.float()], dim=0).to(device) # q, k, v
        if act_aware:
            scaling_diag_matrix = torch.ones(linear1.in_features, device=utils.get_dev())  # avoid zero division
            if hasattr(linear, "scaling_diag_matrix"):
                # logging.info("WARNING: scaling_diag_matrix is used")
                scaling_diag_matrix *= linear1.scaling_diag_matrix.to(utils.get_dev())**alpha
                scaling_diag_matrix += 1e-6  # avoid zero division
                scaling_matrix_inv = None
                w = w * scaling_diag_matrix.view(1, -1)
            elif hasattr(linear, "scaling_diag_matrixS"):
                scaling_diag_matrix = linear1.scaling_diag_matrixS.to(utils.get_dev())
                try:
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                except Exception as e:
                    logging.info("Warning: scaling_diag_matrix is not full rank!")
                    scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0]).to(utils.get_dev())
                    scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
                w = w @ scaling_diag_matrix.float()
        Us = []
        Ss = []
        Vs = []
        try:
            print(rank)
            # SVD decomposition
            U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False)

            # Low rank approximation to the target rank
            U = U[:, :rank]
            S = S[:rank]
            Vt = Vt[:rank, :]
            V = Vt.T
        except Exception as e:
            logging.info(f"Fuse QKV SVD failed for {linear}: {e}")
            logging.info(f"Matrix information: shape={w.size()}, whether contains NaN: {torch.isnan(w).any()}, whether contains Inf: {torch.isinf(w).any()}")
            logging.info(f"Matrix statistics: min={w.min().item()}, max={w.max().item()}, mean={w.mean().item()}, std={w.std().item()}")
            return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )
        if act_aware:
            if scaling_matrix_inv is None:
                V = V / scaling_diag_matrix.view(-1, 1)
            else:
                V =  scaling_matrix_inv.T.float() @ V
        U = U.view(3, -1, rank)
        Us = [U[0], U[1], U[2]]
        Ss = [S]
        Vs = [V]

        if linear.bias is not None:
            biasq = linear.bias.data
            biask = linear1.bias.data
            biasv = linear2.bias.data
        else:
            biasq = None
            biask = None
            biasv = None

        # nan or inf check
        for S in Ss:
            if (S != S).any():
                logging.info("nan in S")
                # For from_linearqkv method return three linear layers
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )
        for U in Us:
            if (U != U).any():
                logging.info("nan in U")
                # For from_linearqkv method return three linear layers
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )
        for V in Vs:
            if (V != V).any():
                logging.info("nan in V")
                # For from_linearqkv method return three linear layers
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )

        assert len(Us)/3 == len(Ss) == len(Vs) == 1
        
        if had_rank:
            new_linearQ = SVDLinear(Us[0], Ss[0], Vs[0], biasq, sigma_fuse, had_K, K, had_mode)
            new_linearK = SVDLinear(Us[1], Ss[0], Vs[0], biask, sigma_fuse, had_K, K, had_mode)
            new_linearV = SVDLinear(Us[2], Ss[0], Vs[0], biasv, sigma_fuse, had_K, K, had_mode)
        else:
            new_linearQ = SVDLinear(Us[0], Ss[0], Vs[0], biasq, sigma_fuse)
            new_linearK = SVDLinear(Us[1], Ss[0], Vs[0], biask, sigma_fuse)
            new_linearV = SVDLinear(Us[2], Ss[0], Vs[0], biasv, sigma_fuse)
        new_linearQ.to(linear.weight.dtype)
        new_linearK.to(linear.weight.dtype)
        new_linearV.to(linear.weight.dtype)
        return new_linearQ.cpu(), new_linearK.cpu(), new_linearV.cpu()


    def forward(self, inp):
        # compute USV^Tx + b
        y = self.BLinear(inp)
        y = self.ALinear(y)
        return y

def from_linearqkv_with_grad(
    linear: nn.Linear,
    linear1: nn.Linear,
    linear2: nn.Linear,
    param_ratio: float,
    act_aware=False,
    ic_split=1,
    oc_split=1,
    alpha=1.,
    sigma_fuse="UV",
    rank_align=1.,
    had_rank=False,
    had_mode='hadamard',
    singular_indices=None,
    seed=0,
):
    global total_rank_sum, total_linear_count
    
    # Directly use pre-calculated SVD results and importance sorting results
    device = utils.get_dev()
    
    # Check whether there are pre-calculated SVD results
    # Fix: Should not get qkv_svd_info from linear.self_attn, but from other places
    # We need to find the self_attn module that contains these linear layers
    
    # Assume these linear layers belong to the same self_attn module
    # Try to get SVD information from parent module
    parent_module = None
    for name, module in linear._modules.items():
        if hasattr(module, 'qkv_svd_info'):
            parent_module = module
            break
    
    # If linear cannot find, try to find in higher layers
    if parent_module is None:
        # Here we need a more general method to find the module that contains qkv_svd_info
        # Temporary solution: Assume these information are stored in the model at a known location
        # For example, can be obtained from the parameters passed in
        if hasattr(linear, 'parent_attn') and hasattr(linear.parent_attn, 'qkv_svd_info'):
            svd_info = linear.parent_attn.qkv_svd_info
        else:
            # If cannot find SVD information, print error and return original linear layer
            logging.info("Error: Cannot find QKV SVD information. Please ensure prepare_qkv_svd function has been called.")
            return linear, linear1, linear2
    else:
        svd_info = parent_module.qkv_svd_info
    
    # Get SVD information
    U = svd_info['U'].to(device)
    S = svd_info['S'].to(device)
    V = svd_info['V'].to(device)
    
    # Use pre-calculated importance sorting results
    if singular_indices is not None:
        logging.info(f"Using {len(singular_indices)} pre-selected singular values")
        
        # Select important singular values and corresponding vectors
        U_selected = U[:, singular_indices]
        S_selected = S[singular_indices]
        V_selected = V[:, singular_indices]
        
        # Update rank to the number of selected singular values
        rank = len(singular_indices)
        
        # logging.info("rank", rank)
        if had_rank:
            utils.set_seed(seed)
            if had_mode == 'hadamard':
                K = 1
                had_K = None
            elif had_mode == 'rh':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'hadamard')
                K = rank
            elif had_mode == 'random':
                had_K = rotation_utils.get_orthogonal_matrix(rank, 'random')
                K = rank
        
        # Update total rank sum and linear layer count
        total_rank_sum += rank
        total_linear_count += 1
        
        logging.info(f"Layer uses rank: {rank}")
        
        # Process activation-aware scaling
        if act_aware and 'scaling_diag_matrix' in svd_info:
            scaling_diag_matrix = svd_info['scaling_diag_matrix'].to(device)
            if scaling_diag_matrix.ndim == 1:
                # ASVD
                # One-dimensional vector, representing diagonal matrix diagonal elements
                V_selected = V_selected / scaling_diag_matrix.view(-1, 1)
            elif scaling_diag_matrix.ndim == 2:
                # SVD-LLM
                # Two-dimensional matrix, complete scaling matrix, need to right multiply inverse matrix
                try:
                    scaling_diag_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
                except RuntimeError as e:
                    logging.info("Warning: scaling_diag_matrix is not full rank, adding epsilon for stability.")
                    eps = 1e-6
                    scaling_diag_matrix += eps * torch.eye(scaling_diag_matrix.shape[0], device=device)
                    scaling_diag_matrix_inv = torch.linalg.inv(scaling_diag_matrix).to(torch.float32)
                
                V_selected = scaling_diag_matrix_inv.T @ V_selected
            
        # Split U into three parts, corresponding to q, k, v
        U_selected = U_selected.view(3, -1, rank)
        Us = [U_selected[0], U_selected[1], U_selected[2]]
        Ss = [S_selected]
        Vs = [V_selected]
        
        # Process bias
        if linear.bias is not None:
            biasq = linear.bias.data
            biask = linear1.bias.data
            biasv = linear2.bias.data
        else:
            biasq = None
            biask = None
            biasv = None
        
        # Check NaN or Inf
        for S in Ss:
            if (S != S).any():
                logging.info("nan in S")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )
        for U in Us:
            if (U != U).any():
                logging.info("nan in U")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )
        for V in Vs:
            if (V != V).any():
                logging.info("nan in V")
                return (
                    nn.Linear(linear.in_features, linear.out_features).to(linear.weight.dtype).to(linear.weight.device),
                    nn.Linear(linear1.in_features, linear1.out_features).to(linear1.weight.dtype).to(linear1.weight.device),
                    nn.Linear(linear2.in_features, linear2.out_features).to(linear2.weight.dtype).to(linear2.weight.device)
                )
        
        assert len(Us)/3 == len(Ss) == len(Vs) == 1
        
        # Create new linear layer
        if had_rank:
            new_linearQ = SVDLinear(Us[0], Ss[0], Vs[0], biasq, sigma_fuse, had_K, K, had_mode)
            new_linearK = SVDLinear(Us[1], Ss[0], Vs[0], biask, sigma_fuse, had_K, K, had_mode)
            new_linearV = SVDLinear(Us[2], Ss[0], Vs[0], biasv, sigma_fuse, had_K, K, had_mode)
        else:
            new_linearQ = SVDLinear(Us[0], Ss[0], Vs[0], biasq, sigma_fuse)
            new_linearK = SVDLinear(Us[1], Ss[0], Vs[0], biask, sigma_fuse)
            new_linearV = SVDLinear(Us[2], Ss[0], Vs[0], biasv, sigma_fuse)
        
        new_linearQ.to(linear.weight.dtype)
        new_linearK.to(linear.weight.dtype)
        new_linearV.to(linear.weight.dtype)
        return new_linearQ.cpu(), new_linearK.cpu(), new_linearV.cpu()

def rsetattr(obj, attr, value):
    """ Recursively set an attribute given a dotted path """
    pre, _, post = attr.rpartition('.')
    if pre:
        obj = rgetattr(obj, pre)  # Get the nested object first
    setattr(obj, post, value)

def rgetattr(obj, attr):
    """ Recursively get an attribute given a dotted path """
    for part in attr.split('.'):
        obj = getattr(obj, part)
    return obj


# @torch.inference_mode()
def svd_vitmm_setup(model, args, tokenizer, image_processor):
    global total_rank_sum, total_linear_count
    # Reset counters
    total_rank_sum = 0
    total_linear_count = 0
   
    if args.act_aware or args.fisher_info or args.grad_info:
        # Load calibration dataset
        logging.info(f"Loading calibration dataset: {args.cal_dataset}")
        calib_loader = data_utils.get_loaders(
            args.cal_dataset, 
            nsamples=args.nsamples,
            seed=args.seed, 
            model=args.model,
            seqlen=model.seqlen, 
            eval_mode=False,
            args=args
        )
        dataloader, _ = calib_loader
    
    # If act_aware is needed, ensure calibration is done first
    if args.act_aware:
        # Perform calibration - directly use the cache function in act_aware_utils
        act_aware_utils.calib_input_distribution(
            model=model, 
            dataloader=dataloader,
            tokenizer=tokenizer, 
            image_processor=image_processor, 
            args=args, 
            method=args.calib_method, 
            use_cache=True,  # Enable cache function
            cache_file=None
        )
        
    model_type = model_utils.get_model_typevit(model)
    # rotate_embeddings(model, Q) # add online rotate❓
    utils.cleanup_memory()
    layers = model_utils.get_vit_layers(model, 
                                    model_type=model_type)
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="vit SVD")):
        full = quant_utils.find_qlayers(layers[idx], layers=[torch.nn.Linear])
        for name, module in full.items():
            if args.kv_fuse:
                if 'k_proj' in name:
                    klinear, vlinear = SVDLinear.from_linearkv(layers[idx].self_attn.k_proj,
                                                    layers[idx].self_attn.v_proj,
                                                    param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                    alpha=args.act_alpha,
                                                    act_aware=args.act_aware,
                                                    rank_align=1.,
                                                    sigma_fuse=args.svd_mode,
                                                    had_rank=args.had_rank,
                                                    had_mode='rh',
                                                    seed=args.seed)
                    rsetattr(layers[idx], 'self_attn.k_proj', klinear)
                    rsetattr(layers[idx], 'self_attn.v_proj', vlinear)
                elif 'k_proj' not in name and 'v_proj' not in name:
                    rsetattr(layers[idx], name, SVDLinear.from_linear(module,
                                                    param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                    alpha=args.act_alpha,
                                                    act_aware=args.act_aware,
                                                    rank_align=1.,
                                                    sigma_fuse=args.svd_mode,
                                                    had_rank=args.had_rank,
                                                    had_mode='rh',
                                                    seed=args.seed))
            else:
                rsetattr(layers[idx], name, SVDLinear.from_linear(module,
                                                    param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                    alpha=args.act_alpha,
                                                    act_aware=args.act_aware,
                                                    rank_align=1.,
                                                    sigma_fuse=args.svd_mode,
                                                    had_rank=args.had_rank,
                                                    had_mode='rh',
                                                    seed=args.seed))
    logging.info("Now finish vit svd")
    layers = model.model.mm_projector
    full = quant_utils.find_qlayers(layers, layers=[torch.nn.Linear])
    for name, module in full.items():
        rsetattr(layers, name, SVDLinear.from_linear(module,
                                                    param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                    alpha=args.act_alpha,
                                                    act_aware=args.act_aware,
                                                    rank_align=1.,
                                                    sigma_fuse=args.svd_mode,
                                                    had_rank=args.had_rank,
                                                    had_mode='rh',
                                                    seed=args.seed))
    logging.info("vit/mm SVD compression completed")
    logging.info(f"vit/mm SVD compression completed with total rank sum: {total_rank_sum} across {total_linear_count} linear layers")
  
# @torch.inference_mode()
def svd_lm_setup(model, args, tokenizer, image_processor):
    global total_rank_sum, total_linear_count
    # Reset counters
    total_rank_sum = 0
    total_linear_count = 0
    
    if args.act_aware or args.fisher_info or args.grad_info or args.latent_smooth:
        # Load calibration dataset
        logging.info(f"Loading calibration dataset: {args.cal_dataset}")
        calib_loader = data_utils.get_loaders(
            args.cal_dataset, 
            nsamples=args.nsamples,
            seed=args.seed, 
            model=args.model,
            seqlen=model.seqlen, 
            eval_mode=False,
            hf_token=args.hf_token, # [FIXME:]for now use this criteria to decide hf-llava-trainfix, now only pass this in gradinfo dataloader
            args = args
        )
        dataloader, _ = calib_loader
        
    # Decide whether to calibrate based on args
    if args.act_aware:
        # Perform calibration - directly use cache function in act_aware_utils
        if args.calib_method == 'cholesky': #  calib_input_distributionlowresources
            logging.info('using low resource version')
            act_aware_utils.calib_input_distributionlowresources(
                model=model, 
                dataloader=dataloader,
                tokenizer=tokenizer, 
                image_processor=image_processor, 
                args=args, 
                method=args.calib_method, 
                use_cache=args.use_cache,  # default enable cache
                cache_file=args.cache_file,
            )
        else:
            act_aware_utils.calib_input_distribution(
                model=model, 
                dataloader=dataloader,
                tokenizer=tokenizer, 
                image_processor=image_processor, 
                args=args, 
                method=args.calib_method, 
                use_cache=args.use_cache,  # default enable cache
                cache_file=args.cache_file,
            )
    
    if args.fisher_info:
        # Perform Fisher information calculation - directly use cache function in fisher_info_utils
        fisher_info_utils.calib_fisher_info(
            model=model,
            dataloader=dataloader,
            tokenizer=tokenizer,
            image_processor=image_processor,
            args=args,
            use_cache=args.use_cache,  # default enable cache
            cache_file=None
        )
    
    if args.grad_info:
        # Perform gradient information calculation - calib_grad_info includes prepare_qkv_svd call
        grad_info_utils.calib_grad_info(
            model=model,
            dataloader=dataloader,
            tokenizer=tokenizer,
            image_processor=image_processor,
            args=args,
            use_cache=args.use_cache,  # default enable cache
            cache_file=args.cache_file
        )
    
    # Continue performing SVD compression
    model_type = model_utils.get_model_type(model)
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    
    if args.grad_info:
        rank, world_size = get_rank_and_world_size()
        top_indices, top_scores, layer_indices_dict = grad_info_utils.svd_qkv_with_grad_info(layers, args, use_cache=args.use_cache, cache_file=args.cache_file) # top_indices: (layer_idx, singular_value_idx)
        for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="LM SVD with grad info")):
            full = quant_utils.find_qlayers(layers[idx], layers=[torch.nn.Linear])
            for name, module in full.items():
                if args.qkv_fuse:
                    if 'q_proj' in name: # fuse QKV
                        # Check whether there are pre-calculated SVD information
                        if not hasattr(layer.self_attn, 'qkv_svd_info'):
                            logging.info(f"Layer {idx} has no pre-calculated SVD information, skipping")
                            continue
                        
                        # Ensure idx is in layer_indices_dict
                        if idx not in layer_indices_dict:
                            logging.info(f"Layer {idx} is not in layer_indices_dict, skipping")
                            continue
                        
                        # Print more debugging information
                        if rank == 0:
                            logging.info(f"Layer {idx} starting QKV fusion SVD, using {len(layer_indices_dict[idx])} singular values")
                        
                        # Temporarily attach SVD information to linear layer, so from_linearqkv can access
                        layer.self_attn.q_proj.parent_attn = layer.self_attn
                        layer.self_attn.k_proj.parent_attn = layer.self_attn
                        layer.self_attn.v_proj.parent_attn = layer.self_attn
                        
                        try:
                            qlinear, klinear, vlinear = from_linearqkv_with_grad(
                                layer.self_attn.q_proj,
                                layer.self_attn.k_proj,
                                layer.self_attn.v_proj,
                                param_ratio=args.rank_ratio,
                                alpha=args.act_alpha,
                                act_aware=args.act_aware,
                                rank_align=1.,
                                sigma_fuse=args.svd_mode,
                                had_rank=args.had_rank,
                                had_mode='random', # ‘rh'
                                singular_indices=layer_indices_dict[idx],
                                seed=args.seed
                            )
                            
                            # Clean up temporary attributes
                            delattr(layer.self_attn.q_proj, 'parent_attn')
                            delattr(layer.self_attn.k_proj, 'parent_attn')
                            delattr(layer.self_attn.v_proj, 'parent_attn')
                            
                            rsetattr(layers[idx], 'self_attn.q_proj', qlinear)
                            rsetattr(layers[idx], 'self_attn.k_proj', klinear)
                            rsetattr(layers[idx], 'self_attn.v_proj', vlinear)
                            if rank == 0:
                                logging.info(f"Layer {idx} QKV fusion SVD completed")
                        except Exception as e:
                            logging.info(f"Layer {idx} QKV fusion SVD failed: {e}")
                            import traceback
                            traceback.logging.info_exc()
                            
                            # Clean up temporary attributes
                            if hasattr(layer.self_attn.q_proj, 'parent_attn'):
                                delattr(layer.self_attn.q_proj, 'parent_attn')
                            if hasattr(layer.self_attn.k_proj, 'parent_attn'):
                                delattr(layer.self_attn.k_proj, 'parent_attn')
                            if hasattr(layer.self_attn.v_proj, 'parent_attn'):
                                delattr(layer.self_attn.v_proj, 'parent_attn')
                                
    else:        
        if args.fisher_info:
            rank_allocation = {} # Store rank allocation percentage for each linear layer
            fisher_sum = 0
            fisher_count = 0
            
            # If using qkv_fuse, need to merge Fisher information of q_proj, k_proj and v_proj
            if args.qkv_fuse:
                # First collect Fisher information of q_proj, k_proj and v_proj for each layer
                layer_qkv_fisher = {}
                for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Sum Fisher Info (QKV Fused)")):
                    full = quant_utils.find_qlayers(layers[idx], layers=[torch.nn.Linear])
                    layer_key = f"layer_{idx}"
                    layer_qkv_fisher[layer_key] = 0
                    
                    for name, module in full.items():
                        if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                            # Add Fisher information of q_proj, k_proj and v_proj together
                            layer_qkv_fisher[layer_key] += module.fisher_info.sum()
                    
                    # Only calculate once per layer (q, k and v merged)
                    if layer_qkv_fisher[layer_key] > 0:
                        fisher_sum += layer_qkv_fisher[layer_key]
                        fisher_count += 1  # Only calculated once per layer, not separately for q, k and v
                
                logging.info(f"Total Fisher Info (QKV Fused): {fisher_sum}, Layer Count: {fisher_count}")
                
                # Calculate rank allocation ratio for each layer
                for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="SVD Rank Allocation (QKV Fused)")):
                    layer_key = f"layer_{idx}"
                    if layer_key in layer_qkv_fisher and fisher_sum > 0:
                        # Calculate the proportion of the layer in total Fisher information
                        layer_ratio = layer_qkv_fisher[layer_key] / fisher_sum * fisher_count
                        # Allocate rank ratio
                        rank_allocation[layer_key] = layer_ratio
                        logging.info(f"Layer {idx} QKV Fused Rank Ratio: {layer_ratio:.4f}")
            elif args.kv_fuse: # If using kv_fuse, need to merge Fisher information of k_proj and v_proj
                # First collect Fisher information of k_proj and v_proj for each layer
                layer_kv_fisher = {}
                for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Sum Fisher Info (KV Fused)")):
                    full = quant_utils.find_qlayers(layers[idx], layers=[torch.nn.Linear])
                    layer_key = f"layer_{idx}"
                    layer_kv_fisher[layer_key] = 0
                    
                    for name, module in full.items():
                        if 'k_proj' in name or 'v_proj' in name:
                            # Add Fisher information of k_proj and v_proj together
                            layer_kv_fisher[layer_key] += module.fisher_info.sum()
                    
                    # Only calculate once per layer (k and v merged)
                    if layer_kv_fisher[layer_key] > 0:
                        fisher_sum += layer_kv_fisher[layer_key]
                        fisher_count += 1  # Only calculated once per layer, not separately for k and v
                
                logging.info(f"Total Fisher Info (KV Fused): {fisher_sum}, Layer Count: {fisher_count}")
                
                # Calculate rank allocation ratio for each layer
                for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="SVD Rank Allocation (KV Fused)")):
                    layer_key = f"layer_{idx}"
                    if layer_key in layer_kv_fisher and fisher_sum > 0:
                        # Calculate the proportion of the layer in total Fisher information
                        layer_ratio = layer_kv_fisher[layer_key] / fisher_sum * fisher_count
                        # Allocate rank ratio
                        rank_allocation[layer_key] = layer_ratio
                        logging.info(f"Layer {idx} KV Fused Rank Ratio: {layer_ratio:.4f}")
            else:
                # Original non-fusion processing logic
                for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Sum Fisher Info")):
                    full = quant_utils.find_qlayers(layers[idx], layers=[torch.nn.Linear])
                    for name, module in full.items():
                        # Only apply SVD compression to k_proj and v_proj
                        if 'k_proj' in name or 'v_proj' in name:
                            fisher_sum += module.fisher_info.sum()
                            fisher_count += 1
                
                logging.info(f"Total Fisher Info: {fisher_sum}")
                
                # Calculate rank allocation ratio for each module
                for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="SVD Rank Allocation")):
                    full = quant_utils.find_qlayers(layers[idx], layers=[torch.nn.Linear])
                    for name, module in full.items():
                        # Only apply SVD compression to k_proj and v_proj
                        if 'k_proj' in name or 'v_proj' in name:
                            # Calculate the proportion of the module in total Fisher information
                            module_ratio = module.fisher_info.sum() / fisher_sum * fisher_count
                            # Allocate rank ratio
                            module_key = f"layer_{idx}_{name}"
                            rank_allocation[module_key] = module_ratio
                            logging.info(f"{module_key} Rank Ratio: {module_ratio:.4f}")
                            
        for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="LM SVD")):
            full = quant_utils.find_qlayers(layers[idx], layers=[torch.nn.Linear])
            for name, module in full.items():
                if args.qkv_fuse:
                    if 'q_proj' in name: # fuse QKV
                        qlinear, klinear, vlinear = SVDLinear.from_linearqkv(layers[idx].self_attn.q_proj,
                                                        layers[idx].self_attn.k_proj,
                                                        layers[idx].self_attn.v_proj,
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        rank_align=1.,
                                                        sigma_fuse=args.svd_mode,
                                                        had_rank=args.had_rank,
                                                        had_mode='random', #'rh'
                                                        seed=args.seed)
                        rsetattr(layers[idx], 'self_attn.q_proj', qlinear)
                        rsetattr(layers[idx], 'self_attn.k_proj', klinear)
                        rsetattr(layers[idx], 'self_attn.v_proj', vlinear)
                elif args.kv_fuse: # Only apply SVD compression to k_proj and v_proj, fuse KV
                    if 'k_proj' in name:
                        klinear, vlinear = SVDLinear.from_linearkv(layers[idx].self_attn.k_proj,
                                                        layers[idx].self_attn.v_proj,
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        rank_align=1.,
                                                        sigma_fuse=args.svd_mode,
                                                        had_rank=args.had_rank,
                                                        had_mode='rh',
                                                        seed=args.seed)
                        rsetattr(layers[idx], 'self_attn.k_proj', klinear)
                        rsetattr(layers[idx], 'self_attn.v_proj', vlinear)
                    elif 'q_proj' in name:
                        rsetattr(layers[idx], name, SVDLinear.from_linear(module,
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}_{name}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        sigma_fuse=args.svd_mode,
                                                        rank_align=1.,
                                                        had_rank=False,
                                                        had_mode='rh',
                                                        seed=args.seed))
                    elif 'q_proj' in name:
                        rsetattr(layers[idx], name, SVDLinear.from_linear(module,
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}_{name}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        sigma_fuse=args.svd_mode,
                                                        rank_align=1.,
                                                        had_rank=False,
                                                        had_mode='rh',
                                                        seed=args.seed))
                else:
                    # if 'k_proj' in name or 'v_proj' in name or 'q_proj' in name: # Only apply SVD compression to k_proj and v_proj
                    if 'k_proj' in name or 'v_proj' in name or 'q_proj' in name: # Only apply SVD compression to k_proj and v_proj
                        rsetattr(layers[idx], name, SVDLinear.from_linear(module,
                                                        param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}_{name}"] if args.fisher_info else args.rank_ratio,
                                                        alpha=args.act_alpha,
                                                        act_aware=args.act_aware,
                                                        sigma_fuse=args.svd_mode,# if 'k_proj' in name else "U",
                                                        rank_align=1.,
                                                        had_rank=False,
                                                        had_mode='rh',
                                                        seed=args.seed))

    if args.had_rank:
        logging.info("Starting to apply had_rank")
        rotation_utils.had_transform_rank(model)

    logging.info(f"Language model SVD compression completed with total rank sum: {total_rank_sum} across {total_linear_count} linear layers")

def cap_and_redistribute_rank_allocation(rank_allocation: dict, cap_ratio: float) -> dict:
    """
    Cap each layer's rank ratio at cap_ratio, and redistribute excess proportionally
    to layers that were not capped.

    Args:
        rank_allocation (dict): layer_name -> raw rank ratio
        cap_ratio (float): maximum allowed ratio per layer (e.g., 1.0)

    Returns:
        dict: layer_name -> capped and redistributed rank ratio
    """
    raw_ratios = rank_allocation.copy()
    clipped_ratios = {}
    excess = 0.0

    # First pass: clip ratios and accumulate excess
    for key, ratio in raw_ratios.items():
        if ratio > cap_ratio:
            clipped_ratios[key] = cap_ratio
            excess += ratio - cap_ratio
        else:
            clipped_ratios[key] = ratio

    # Get layers eligible to receive redistribution
    redistribute_keys = [k for k in raw_ratios if raw_ratios[k] <= cap_ratio]
    redistribute_total = sum(raw_ratios[k] for k in redistribute_keys)

    # Redistribute the excess
    for k in redistribute_keys:
        if redistribute_total > 0:
            share = raw_ratios[k] / redistribute_total
            clipped_ratios[k] += share * excess

    return clipped_ratios

def cap_rank_allocation(rank_allocation: dict, cap_ratio: float) -> dict:
    """
    Cap each layer's rank ratio at cap_ratio, and redistribute excess proportionally
    to layers that were not capped.

    Args:
        rank_allocation (dict): layer_name -> raw rank ratio
        cap_ratio (float): maximum allowed ratio per layer (e.g., 1.0)

    Returns:
        dict: layer_name -> capped and redistributed rank ratio
    """
    raw_ratios = rank_allocation.copy()
    clipped_ratios = {}
    excess = 0.0

    # First pass: clip ratios and accumulate excess
    for key, ratio in raw_ratios.items():
        if ratio > cap_ratio:
            clipped_ratios[key] = cap_ratio

    return clipped_ratios



def svd_llava_setup(model, args, tokenizer=None, image_processor=None):
    if args.act_aware or args.fisher_info or args.grad_info:
        # Load calibration dataset
        logging.info(f"Loading calibration dataset: {args.cal_dataset}")
        calib_loader = data_utils.get_loaders(
            args.cal_dataset, 
            nsamples=args.nsamples,
            seed=args.seed, 
            model=args.model,
            seqlen=model.seqlen, 
            eval_mode=False,
            args=args
        )
        dataloader, _ = calib_loader
        
    # Decide whether to calibrate based on args
    if args.act_aware:
        # Perform calibration - directly use cache function in act_aware_utils
        act_aware_utils.calib_input_distribution(
            model=model, 
            dataloader=dataloader,
            tokenizer=tokenizer, 
            image_processor=image_processor, 
            args=args, 
            method=args.calib_method, 
            use_cache=True,  # default enable cache
            cache_file=None
        )
    
    # Continue performing SVD compression
    model_type = model_utils.get_model_type(model)
    utils.cleanup_memory()
    layers = model_utils.get_transformer_layers(model, model_type=model_type)
    
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="LM SVD")):
        full = quant_utils.find_qlayers(layers[idx], layers=[torch.nn.Linear])
        for name, module in full.items():
            if args.qkv_fuse:
                if 'q_proj' in name: # fuse QKV
                    qlinear, klinear, vlinear = SVDLinear.from_linearqkv(layers[idx].self_attn.q_proj,
                                                    layers[idx].self_attn.k_proj,
                                                    layers[idx].self_attn.v_proj,
                                                    param_ratio=args.rank_ratio * rank_allocation[f"layer_{idx}"] if args.fisher_info else args.rank_ratio,
                                                    alpha=args.act_alpha,
                                                    act_aware=args.act_aware,
                                                    rank_align=1.,
                                                    sigma_fuse=args.svd_mode,
                                                    had_rank=args.had_rank,
                                                    had_mode='rh',
                                                    seed=args.seed)
                    rsetattr(layers[idx], 'self_attn.q_proj', qlinear)
                    rsetattr(layers[idx], 'self_attn.k_proj', klinear)
                    rsetattr(layers[idx], 'self_attn.v_proj', vlinear)
            # Only apply SVD compression to k_proj and v_proj
            elif args.kv_fuse:
                if 'k_proj' in name:
                    klinear, vlinear = SVDLinear.from_linearkv(layers[idx].self_attn.k_proj,
                                                    layers[idx].self_attn.v_proj,
                                                    param_ratio=args.rank_ratio,
                                                    alpha=args.act_alpha,
                                                    act_aware=args.act_aware,
                                                    rank_align=1.,
                                                    sigma_fuse=args.svd_mode,
                                                    had_rank=args.had_rank,
                                                    had_mode='rh',
                                                    seed=args.seed)
                    rsetattr(layers[idx], 'self_attn.k_proj', klinear)
                    rsetattr(layers[idx], 'self_attn.v_proj', vlinear)
            else:
                if 'k_proj' in name or 'v_proj' in name:
                    rsetattr(layers[idx], name, SVDLinear.from_linear(module,
                                                    param_ratio=args.rank_ratio,
                                                    alpha=args.act_alpha,
                                                    act_aware=args.act_aware,
                                                    rank_align=1.,
                                                    sigma_fuse=args.svd_mode,
                                                    had_rank=args.had_rank,
                                                    had_mode='rh',
                                                    seed=args.seed))
    
    logging.info("Language model SVD compression completed")
