import torch
import torch.nn as nn
import torch.distributed as dist
import logging
import model_utils
class weightLearner(nn.Module):
    def __init__(self, hidden_dim, affine=False, bias=None):
        super(weightLearner, self).__init__()
        self.affine = affine
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        if bias is not None:
            self.linear.bias.data = nn.Parameter(bias)
            self.linear.weight.data = nn.Parameter(torch.zeros_like(self.linear.weight.data))
    def forward(self, x):
        return self.linear(x) + x


import torch.nn.init as init
class weightLearnerlowrank(nn.Module):
    def __init__(self, hidden_dim, affine=False, bias=None, rank=128):
        super(weightLearnerlowrank, self).__init__()
        self.affine = affine
        self.linear_down = nn.Linear(hidden_dim, rank, bias=False)
        self.linear = nn.Linear(rank, hidden_dim, bias=True)
        init.kaiming_normal_(self.linear_down.weight)
        init.zeros_(self.linear.weight)
        if bias is not None:
            self.linear.bias.data = nn.Parameter(bias)
            self.linear.weight.data = nn.Parameter(torch.zeros_like(self.linear.weight.data))
    def forward(self, x):
        return self.linear(self.linear_down(x)) + x

def train_bias_linear(input_fp16, input_fuse, num_epochs=100, lr=0.5, Q=None, args=None, weight=None, bias=None, linear=None):
    import time
    torch.set_grad_enabled(True)
    rank= dist.get_rank() if dist.is_initialized() else 0
    print(torch.is_grad_enabled())  # True means gradients are enabled, False means they are disabled
    nsamples = args.nsamples # to handle v1.6/smovlm etc dynamic resolution
    if isinstance(input_fp16, list):
        input_fp16 = torch.cat(input_fp16, dim=0) # sum(bs_i) , token, dim
        input_fuse = torch.cat(input_fuse, dim=0) # sum(bs_i) , token, dim
        nsamples = input_fp16.shape[0]
    input_fp16 = input_fp16.cuda().float()  # Ensure both inputs are float32 and moved to GPU
    input_fuse = input_fuse.cuda().float()  # Ensure input_fuse is also on the GPU
    if Q is not None:
        input_fp16 = input_fp16 @ Q.float()
    # Initialize the model (BiasLearner)
    # model = weightLearnerv2(input_fp16.shape[-1], affine=weight, bias=bias).cuda()
    model = weightLearner(input_fp16.shape[-1], affine=weight, bias=None).cuda()
    if linear is not None:
        lineary7, input_fp16 = linear['linear'].cuda().float() , linear['target']
    # Define the optimizer (Adam in this case)
    if rank == 0:
        logging.info(f"Using SGD, LR: {lr}")
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[_ for _ in range(500, num_epochs, 500)], gamma=0.5)
    loss_fn = nn.MSELoss()  # Use MSELoss 
    # loss_fn = nn.L1Loss() # Use MAELoss
    loss_curve = []
    # Training loop
    start_time = time.time()
    if args.bs != nsamples and not args.bs_to_nsamples:
        args.bs = nsamples
        logging.info(f"bs is neq than nsamples, assign to {args.bs}")
    else:
        logging.info(f"bs remain {args.bs}")
    for epoch in range(num_epochs):
        num_acc = nsamples//args.bs
        optimizer.zero_grad()
        for bs_start in range(0, nsamples, args.bs):
            bs_end = min(bs_start + args.bs, nsamples)
            # Forward pass: Compute prediction (input_fuse + bias)
            pred = model(input_fuse[bs_start:bs_end])  # pred will have requires_grad=True because bias is a learnable parameter
            
            # Compute the loss (Mean Squared Error between input_fp16 and pred)
            # loss = loss_fn(pred, input_fp16)
            if linear is not None:
                pred = lineary7(pred)
                loss = loss_fn(pred, input_fp16[bs_start:bs_end].float())/ num_acc
            else:
                loss = loss_fn(pred, input_fp16[bs_start:bs_end])/ num_acc
            # loss = loss_fn(pred, input_fp16[bs_start:bs_end]) / input_fp16[bs_start:bs_end].abs().mean()
            # loss = ((pred-input_fp16)**2).sum(-1).mean()

            # Backpropagation: Calculate gradients with respect to the bias parameter
            loss.backward()

            # Update parameters (bias)
            if bs_end == nsamples:
                optimizer.step()
                optimizer.zero_grad()
        # scheduler.step()
        if torch.isnan(loss):
            continue
        loss_curve.append(loss.item())
        # Print loss every 100 epochs
        if epoch % 100 == 0:
            if rank == 0:
                logging.info(f"Epoch {epoch}, Loss: {loss.item()}, LR: {scheduler.get_last_lr()[0]:.2f}")
    # if rank == 0:
    logging.info(f"Epoch {epoch}, Loss: {loss.item()}")
    logging.info(f"Training completed in {time.time() - start_time:.3f} seconds.")
    logging.info(f"Norm: {input_fp16.abs().mean().item()}")
    def save_plot(loss_curve, name=None):
        import matplotlib.pyplot as plt
        import os
        save_path = args.save_path + '/save_loss/'
        os.makedirs(save_path, exist_ok=True)
        plt.figure(figsize=(10, 5))
        epochs = [_ for _ in range(num_epochs)]
        plt.plot(epochs, loss_curve, label = 'loss', linewidth=2)
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.title(f"Loss curve-loss")
        plt.savefig(f'{save_path}lr_{lr}_ep_{num_epochs}sgd_{name}.png')
        plt.close()
    if rank == 0:
        save_plot(loss_curve, 'linear')
    del optimizer
    return model  # Return the learned linear layer

def fuse_linearbias(bias, model, Q=None, is_mm = False, args=None):
    linear = bias
    if not is_mm:
        logging.info("not implementing fusing laffinebeta to mlp.fc2!")
    else:
        if args is not None and args.svd_vit:
            if type(model) == model_utils.LLAVA_NEXT_HF:
                mmfc1 = model.multi_modal_projector.linear_1.BLinear
            else:
                mmfc1 = model.model.mm_projector[0].BLinear
        else:
            if type(model) == model_utils.LLAVA_NEXT_HF:
                mmfc1 = model.multi_modal_projector.linear_1
            else:
                mmfc1 = model.model.mm_projector[0]
        mmfc1.fuse_linear = True
        mmfc1.bias_param = linear
        logging.info("finish fusing linearbeta to mmprojector actquantwrapper!")