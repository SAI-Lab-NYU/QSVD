import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import utils
import gptq_utils
import data_utils
import quant_utils
import model_utils
from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
import logging

import os
import torch.distributed as dist
import datetime

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

import torch

def insert_ignore_index_after_prompt(input_ids, output_ids, image_token_id=32000, ignore_index=-100):
    """
    In output_ids, after the prompt part and before the image token part,
    insert the corresponding number of ignore_index (-100) for masking during loss calculation.

    Args:
        input_ids (torch.Tensor): shape (seq_len,)
        output_ids (torch.Tensor): shape (seq_len,)
        image_token_id (int): image placeholder token id, default 32000
        ignore_index (int): marker to be ignored by CrossEntropyLoss, default -100

    Returns:
        torch.Tensor: processed output_ids with ignore_index segment
    """
    # Find the position of the first <image>
    image_positions = (input_ids == image_token_id).nonzero(as_tuple=True)
    if len(image_positions[0]) == 0:
        # No image token, return original output_ids
        return output_ids.clone()

    first_image_idx = image_positions[0][0].item()
    num_image_tokens = (input_ids == image_token_id).sum().item()

    # Split prompt and remaining parts
    prompt_output_ids = output_ids[:first_image_idx]
    rest_output_ids = output_ids[first_image_idx:]

    # Construct ignore_index segment
    ignore_prefix = torch.full((num_image_tokens,), ignore_index, dtype=output_ids.dtype, device=output_ids.device)

    # Concatenate
    final_output_ids = torch.cat([prompt_output_ids, ignore_prefix, rest_output_ids], dim=0)

    return final_output_ids

@torch.enable_grad()
def calib_grad_info(model, dataloader, tokenizer, image_processor, args, use_cache=True, cache_file=None):
    """
    Calculate Grad matrix for each layer of the model to evaluate parameter importance
    
    Args:
        model: Model to be calibrated
        tokenizer: Tokenizer
        image_processor: Image processor
        args: Parameter configuration
        use_cache: Whether to use cache
        cache_file: Cache file path, automatically generated if None
    """
    model_id = model.config._name_or_path
    
    if cache_file is None:
        cache_dir = "cache"
        if args.cache_in_log:
            cache_dir = args.save_path + "/cache"
        os.makedirs(cache_dir, exist_ok=True)
        # Add relevant information to cache file name
        calib_method_info = args.calib_method if hasattr(args, "act_aware") and args.act_aware else "no_act_aware"
        # cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{rotate_info}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma_grad_info.pt")
        if args.a_clip_ratio == 1.0:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma_grad_info.pt")
        else:
            cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_aclip{args.a_clip_ratio}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma_grad_info.pt")
    else:
        calib_method_info = args.calib_method if hasattr(args, "act_aware") and args.act_aware else "no_act_aware"
        cache_file = os.path.join(args.cache_file, f"{args.model.replace('/','_')}_{args.nsamples}_{calib_method_info}_sigma_grad_info.pt")
        # if args.a_clip_ratio == 1.0:
        #     cache_file = os.path.join(args.cache_file, f"{args.model.replace('/','_')}_{args.nsamples}_{calib_method_info}_sigma_grad_info.pt")
        # else:
        #     cache_file = os.path.join(args.cache_file, f"{args.model.replace('/','_')}_aclip{args.a_clip_ratio}_{args.nsamples}_{calib_method_info}_sigma_grad_info.pt")
    # First perform QKV SVD decomposition and store
    logging.info('start qkv svd for grad')
    prepare_qkv_svd(model, args)
    logging.info('finish qkv svd for grad')


    if os.path.exists(cache_file) and use_cache:
        logging.info(f"Loading Grad information cache from {cache_file}...")
        all_grad_info = torch.load(cache_file, map_location="cpu")
        # Load gradient information into the self_attn.S_grad_info attribute of corresponding layers
        for idx, layer in enumerate(model_utils.get_layers(model)):
            layer_key = f"layer_{idx}"
            if layer_key in all_grad_info:
                layer.self_attn.S_grad_info = all_grad_info[layer_key].to(utils.get_dev())
        logging.info("Successfully loaded Grad information cache!")
        return
    
    print("Starting Grad information calculation...")
    logging.info('start grad computing')
    model.eval()

    #batch_input_ids, batch_images, batch_output_ids = data_utils.process_data(dataloader, image_processor, model, tokenizer)

    # Ensure the entire model is on CUDA
    device = utils.get_dev()
    model = model.to(device)

    accumulation_steps = 1   # Number of accumulated batches
    batch_count = 0          # Accumulated batch counter
    
    # Set model to training mode and only allow gradient computation for q_proj, k_proj, v_proj layers
    model.train()
    for name, param in model.named_parameters():
        if 'model.layers' in name:
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False

    for batch in tqdm(dataloader, desc="Computing Gradient Information"):
        try:
            if tokenizer is None: # SmolVLM
                    # Use message_to_prompt_train to process batch data
                inputs, _, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                # inputs, _, outputs
                # Define recursive function to move nested tensor structures to specified device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, target_device) for item in obj)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    else:
                        return obj
                
                # Move data to corresponding device
                inputs = move_to_device(inputs, device)
                # outputs = move_to_device(outputs, device)
                output_ids = move_to_device(output_ids, device)

                input_ids = inputs.get('input_ids')
                output_ids = output_ids 
                
                # Adjust input and label lengths to match
                # breakpoint()
                if input_ids.size(1) != output_ids.size(1):
                    max_len = max(input_ids.size(1), output_ids.size(1))
                    if input_ids.size(1) < max_len:
                        padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                            dtype=input_ids.dtype, device=input_ids.device)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                    else:
                        input_ids = input_ids[:, :max_len]
                    if output_ids.size(1) < max_len:
                        padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                            fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                        output_ids = torch.cat([output_ids, padding], dim=1)
                    else:
                        output_ids = output_ids[:, :max_len]
                    # print(f"Adjusted input and label lengths to {max_len}")
                    input_ids = input_ids.to(device)
                    output_ids = output_ids.to(device)
                
                inputs['input_ids'] = input_ids
                inputs['attention_mask'] = input_ids.ne(0).to(device)

                #breakpoint()

                # Calculate loss for current batch and perform gradient accumulation
                with torch.enable_grad():
                    outputs = model(**inputs, labels=output_ids)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()
            elif tokenizer == 'hf_v16': # LLaVA Next
                    # Use message_to_prompt_train to process batch data
                inputs, _, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                
                # Define recursive function to move nested tensor structures to specified device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, target_device) for item in obj)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    else:
                        return obj
                
                # Move data to corresponding device
                inputs = move_to_device(inputs, device)
                output_ids = move_to_device(output_ids, device)
                
                # Adjust input and label lengths to match
                # input_ids = inputs.get('input_ids')[:, :-1]
                # output_ids = inputs.get('input_ids')[:, 1:]
                # input_ids = inputs.get('input_ids')
                # output_ids = outputs.get('input_ids')
                
                # final_output_ids = insert_ignore_index_after_prompt(input_ids[0],output_ids[0], image_token_id=32000, ignore_index=-100) # This image_token_id is for LLaVANextProcessor
                
                # # Create a new output_ids tensor with size matching final_output_ids
                # new_output_ids = torch.full((output_ids.size(0), final_output_ids.size(0)), 
                #                           fill_value=-100, 
                #                           dtype=output_ids.dtype, 
                #                           device=output_ids.device)
                # # Assign final_output_ids to the first sample
                # new_output_ids[0] = final_output_ids
                # # Replace original output_ids
                # output_ids = new_output_ids

                input_ids = inputs.get('input_ids')
                output_ids = output_ids

                # breakpoint()
                # 0 as pad token id, following many tokenizers
                if input_ids.size(1) != output_ids.size(1):
                    max_len = max(input_ids.size(1), output_ids.size(1))
                    if input_ids.size(1) < max_len:
                        padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                            dtype=input_ids.dtype, device=input_ids.device)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                    else:
                        input_ids = input_ids[:, :max_len]
                    if output_ids.size(1) < max_len:
                        padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                            fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                        output_ids = torch.cat([output_ids, padding], dim=1)
                    else:
                        output_ids = output_ids[:, :max_len]
                    # print(f"Adjusted input and label lengths to {max_len}")
                    input_ids = input_ids.to(device)
                    output_ids = output_ids.to(device)
                
                inputs['input_ids'] = input_ids
                inputs['attention_mask'] = input_ids.ne(0).to(device)

                #breakpoint()

                # Calculate loss for current batch and perform gradient accumulation
                with torch.enable_grad():
                    outputs = model(**inputs, labels=output_ids)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()
            elif 'hf_v16' in str(tokenizer): # handle 'hf_v16_trainfix'
                # print('now using trainfix')
                inputs, _, _ = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                
                # Define recursive function to move nested tensor structures to specified device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, target_device) for item in obj)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    else:
                        return obj
                
                # Move data to corresponding device
                inputs = move_to_device(inputs, device)
                
                # Adjust input and label lengths to match
                input_ids = inputs.get('input_ids')
                output_ids = inputs.get('labels')
                if input_ids.size(1) != output_ids.size(1):
                    max_len = max(input_ids.size(1), output_ids.size(1))
                    if input_ids.size(1) < max_len:
                        padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                            dtype=input_ids.dtype, device=input_ids.device)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                    else:
                        input_ids = input_ids[:, :max_len]
                    if output_ids.size(1) < max_len:
                        padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                            fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                        output_ids = torch.cat([output_ids, padding], dim=1)
                    else:
                        output_ids = output_ids[:, :max_len]
                    print(f"Adjusted input and label lengths to {max_len}")
                    input_ids = input_ids.to(device)
                    output_ids = output_ids.to(device)
                    inputs['labels'] = output_ids
                else:
                    print(f"Input and label length {input_ids.size(1)}")
                if args.token_length >0:
                    input_ids = input_ids[:,:args.token_length]
                    output_ids = output_ids[:,:args.token_length]
                    inputs['labels'] = output_ids
                    print(f'truncate inputs to token length{input_ids.size(1)}')

                inputs['input_ids'] = input_ids
                inputs['attention_mask'] = input_ids.ne(0).to(device)
                del output_ids
                del input_ids
                        
                # Calculate loss for current batch and perform gradient accumulation
                with torch.enable_grad():
                    outputs = model(**inputs)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()

            else: # LLaVA
                # Use message_to_prompt_train to process batch data
                input_ids, images, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer)
                
                # Define recursive function to move nested tensor structures to specified device
                def move_to_device(obj, target_device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(target_device)
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(move_to_device(item, target_device) for item in obj)
                    elif isinstance(obj, dict):
                        return {k: move_to_device(v, target_device) for k, v in obj.items()}
                    else:
                        return obj
                
                # Move data to corresponding device
                input_ids = move_to_device(input_ids, device)
                image_sizes = None
                if images is not None:
                    images, image_sizes = images
                    images = move_to_device(images, device)
                output_ids = move_to_device(output_ids, device)
                
                # input_ids = input_ids[:, :-1]
                # output_ids = input_ids[:, 1:].clone()
                # output_ids[output_ids == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
                # Adjust input and label lengths to match
                if input_ids.size(1) != output_ids.size(1):
                    max_len = max(input_ids.size(1), output_ids.size(1))
                    if input_ids.size(1) < max_len:
                        padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), 
                                            dtype=input_ids.dtype, device=input_ids.device)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                    else:
                        input_ids = input_ids[:, :max_len]
                    if output_ids.size(1) < max_len:
                        padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), 
                                            fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                        output_ids = torch.cat([output_ids, padding], dim=1)
                    else:
                        output_ids = output_ids[:, :max_len]
                    # print(f"Adjusted input and label lengths to {max_len}")
                    input_ids = input_ids.to(device)
                    output_ids = output_ids.to(device)
                
                attention_mask = input_ids.ne(0).to(device) 
                # breakpoint()
                # Calculate loss for current batch and perform gradient accumulation
                with torch.enable_grad():
                    outputs = model(input_ids=input_ids, images=images, labels=output_ids, attention_mask=attention_mask, image_sizes=image_sizes)
                    loss = outputs[0]
                    loss = loss / accumulation_steps  # Normalize loss
                    loss.backward()
            
            batch_count += 1  # Count each batch
            
            # Perform backpropagation when accumulated the set number of batches
            if batch_count % accumulation_steps == 0:
                # Update gradient information for S in each layer
                for idx, layer in enumerate(model_utils.get_layers(model)):
                    if hasattr(layer.self_attn, 'qkv_svd_info'):
                        svd_info = layer.self_attn.qkv_svd_info
                        q_linear = layer.self_attn.q_proj
                        k_linear = layer.self_attn.k_proj
                        v_linear = layer.self_attn.v_proj
                        
                        if (q_linear.weight.grad is not None and 
                            k_linear.weight.grad is not None and 
                            v_linear.weight.grad is not None):
                            grad_cat = torch.cat([
                                q_linear.weight.grad.detach().to(torch.float32),
                                k_linear.weight.grad.detach().to(torch.float32),
                                v_linear.weight.grad.detach().to(torch.float32)
                            ], dim=0).to(device)
                            
                            if args.act_aware:
                                scaling_matrix_inverse_transpose = svd_info['scaling_matrix_inverse_transpose'].to(device)
                                if scaling_matrix_inverse_transpose.ndim == 1:
                                    # 1D vector representing diagonal matrix elements
                                    grad_cat = grad_cat * scaling_matrix_inverse_transpose.view(1, -1).to(torch.float32)  # Scale each column
                                elif scaling_matrix_inverse_transpose.ndim == 2:
                                    # 2D matrix representing full scaling matrix (possibly non-diagonal)
                                    grad_cat = grad_cat @ scaling_matrix_inverse_transpose.to(torch.float32)  # Right multiply matrix
                
                            U = svd_info['U'].to(device).to(torch.float32)
                            V = svd_info['V'].to(device).to(torch.float32)
                            # S_grad = torch.diag(U.T @ grad_cat @ V)
                            # more efficient - compute only diagonal elements
                            S_grad = torch.sum(U * (grad_cat @ V), dim=0)
                            S_grad_squared = S_grad.pow(2)
                            
                            if not hasattr(layer.self_attn, 'S_grad_info'):
                                layer.self_attn.S_grad_info = S_grad_squared
                            else:
                                layer.self_attn.S_grad_info += S_grad_squared
                
                model.zero_grad()  # Clear gradients
        
        except Exception as e:
            print(f"Error occurred during Grad information calculation: {e}")
            import traceback
            print("Detailed error information:")
            traceback.print_exc()
            if isinstance(batch, dict):
                print(f"Batch data keys: {list(batch.keys())}")
            elif isinstance(batch, list) and len(batch) > 0:
                print(f"Type of first item in batch data: {type(batch[0])}")
            continue

    # Normalize S gradient information
    if batch_count > 0:
        for layer in model_utils.get_layers(model):
            if hasattr(layer.self_attn, 'S_grad_info'):
                layer.self_attn.S_grad_info = layer.self_attn.S_grad_info.div(batch_count//accumulation_steps).sqrt()

    logging.info('finished grad computing')
    # Save S gradient information
    all_grad_info = {}
    for idx, layer in enumerate(model_utils.get_layers(model)):
        if hasattr(layer.self_attn, 'S_grad_info'):
            print(f"Layer {idx}: {layer.self_attn.S_grad_info.shape}")
            all_grad_info[f"layer_{idx}"] = layer.self_attn.S_grad_info.cpu()

    logging.info(f"Saving Grad information cache to {cache_file}...")
    torch.save(all_grad_info, cache_file)
    logging.info("Grad information cache saved successfully!")

def prepare_qkv_svd(model, args):
    """
    Pre-process QKV layers with SVD decomposition and store results in attention modules
    
    Args:
        model: Model to be processed
        args: Parameter configuration
    """
    print("Preprocessing QKV layer SVD decomposition...")
    device = utils.get_dev()
    alpha = args.act_alpha
    # model_utils.get_layers(model)
    
    for idx, layer in enumerate(tqdm(model_utils.get_layers(model), desc="Preparing QKV SVD")):
        q_linear = layer.self_attn.q_proj
        k_linear = layer.self_attn.k_proj
        v_linear = layer.self_attn.v_proj
        
        try:
            # Move weights to CUDA device
            w = torch.cat([
                q_linear.weight.data.float(), 
                k_linear.weight.data.float(), 
                v_linear.weight.data.float()
            ], dim=0).to(device)
            
            # Apply activation-aware scaling (if enabled)
            if args.act_aware:
                scaling_diag_matrix = torch.ones(k_linear.in_features, device=utils.get_dev())  # avoid zero division
                if hasattr(k_linear, "scaling_diag_matrix"):
                    # print("WARNING: scaling_diag_matrix is used")
                    scaling_diag_matrix *= (k_linear.scaling_diag_matrix.to(utils.get_dev())).to(torch.float32)**alpha
                    scaling_diag_matrix += 0  # avoid zero division
                    w = w * scaling_diag_matrix.view(1, -1)
                    scaling_matrix_inverse_transpose = scaling_diag_matrix**(-1) # (k_linear.scaling_diag_matrix.to(utils.get_dev())).to(torch.float32)**(-alpha)
                elif hasattr(k_linear, "scaling_diag_matrixS"):
                    scaling_diag_matrix = k_linear.scaling_diag_matrixS.to(utils.get_dev())
                    w = w @ scaling_diag_matrix.float()
                    scaling_matrix_inverse_transpose = torch.linalg.inv(scaling_diag_matrix).transpose(-1, -2)
                
            # SVD decomposition
            U, S, Vt = torch.linalg.svd(w.to(torch.float32), full_matrices=False) # SVD decomposition of WS
            V = Vt.T
            
            # Store SVD results
            layer.self_attn.qkv_svd_info = {
                'U': U.cpu(),
                'S': S.cpu(),
                'V': V.cpu()
            } 
            
            if hasattr(args, "act_aware") and args.act_aware:
                layer.self_attn.qkv_svd_info['scaling_diag_matrix'] = scaling_diag_matrix.cpu()
                layer.self_attn.qkv_svd_info['scaling_matrix_inverse_transpose'] = scaling_matrix_inverse_transpose.cpu()
            
            print(f"Layer {idx} QKV SVD completed, S shape: {S.shape}")
            
        except Exception as e:
            print(f"Layer {idx} QKV SVD failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("QKV layer SVD preprocessing completed")

def svd_qkv_with_grad_info(layers, args, use_cache=True, cache_file=None):
    """
    Perform SVD decomposition on QKV layer fusion and utilize gradient information to construct S importance scores
    
    Args:
        layers: List of model layers
        args: Parameter configuration
        use_cache: Whether to use cache
        cache_file: Cache file path, automatically generated if None
    
    Returns:
        grad_scores_dict: Dictionary containing gradient importance scores for S in each layer
    """
    grad_alpha = args.grad_alpha
    # Automatically generate cache file path
    if cache_file is None:
        cache_dir = "cache"
        if hasattr(args, "cache_in_log") and args.cache_in_log:
            cache_dir = args.save_path + "/cache"
        os.makedirs(cache_dir, exist_ok=True)
        # Add relevant information to cache file name
        calib_method_info = args.calib_method if hasattr(args, "act_aware") and args.act_aware else "no_act_aware"
        cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{args.nsamples}_{args.seed}_{calib_method_info}_{grad_alpha}_sigma_grad_scores.pt")
    else:
        calib_method_info = args.calib_method if hasattr(args, "act_aware") and args.act_aware else "no_act_aware"
        cache_file = os.path.join(args.cache_file, f"{args.model.replace('/','_')}_{args.nsamples}_{calib_method_info}_{grad_alpha}_sigma_grad_scores.pt")

    # If cache exists and cache is enabled, load directly
    if os.path.exists(cache_file) and use_cache:
        logging.info(f"Loading gradient importance score cache from {cache_file}...")
        grad_scores_dict = torch.load(cache_file, map_location="cpu")
        logging.info("Successfully loaded gradient importance score cache!")
        
    else:
        # Load gradient information cache file
        grad_info_cache_dir = "cache"
        if hasattr(args, "cache_in_log") and args.cache_in_log:
            grad_info_cache_dir = args.save_path + "/cache"
        
        # Build gradient information cache file path
        if hasattr(args, "a_clip_ratio") and args.a_clip_ratio == 1.0:
            grad_info_cache = os.path.join(grad_info_cache_dir, f"{args.model.replace('/','_')}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma_grad_info.pt")
        else:
            grad_info_cache = os.path.join(grad_info_cache_dir, f"{args.model.replace('/','_')}_aclip{args.a_clip_ratio}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma_grad_info.pt")
        
        # Check if gradient information cache exists
        if os.path.exists(grad_info_cache):
            logging.info(f"Loading gradient information from {grad_info_cache}...")
            all_grad_info = torch.load(grad_info_cache, map_location="cpu")
            
            # Load gradient information into corresponding layers
            for idx, layer in enumerate(layers):
                layer_key = f"layer_{idx}"
                if layer_key in all_grad_info:
                    if not hasattr(layer.self_attn, 'S_grad_info'):
                        layer.self_attn.S_grad_info = all_grad_info[layer_key].to(utils.get_dev())
            logging.info("Successfully loaded gradient information!")
        
        # Directly use pre-computed S gradient information
        grad_scores_dict = {}
        
        for idx, layer in enumerate(layers):
            if hasattr(layer.self_attn, 'qkv_svd_info') and hasattr(layer.self_attn, 'S_grad_info'):
                svd_info = layer.self_attn.qkv_svd_info
                S = svd_info['S']
                S_grad = layer.self_attn.S_grad_info
                
                # Ensure S and S_grad are on the same device (both moved to CUDA)
                device = utils.get_dev()  # Get CUDA device
                S = svd_info['S'].to(device).to(torch.float16)
                S_grad = layer.self_attn.S_grad_info.to(device).to(torch.float16)
                
                # Calculate importance score: |S| * |S_grad|
                importance_score = torch.abs(S) * (torch.abs(S_grad)**grad_alpha)
                
                # Move result back to CPU for saving
                layer_key = f"layer_{idx}"
                grad_scores_dict[layer_key] = importance_score.cpu()
                
                print(f"Layer {idx} importance score computed, shape: {importance_score.shape}")
            else:
                print(f"Warning: Layer {idx} lacks necessary SVD information or gradient information, cannot compute importance score") 
        
        # Save gradient importance score cache
        logging.info(f"Saving gradient importance score cache to {cache_file}...")
        torch.save(grad_scores_dict, cache_file)
        logging.info("Gradient importance score cache saved successfully!")
    
    # Get indices and scores of top k important singular values
    num_layers = len(layers)
    hidden_size = layers[0].self_attn.q_proj.in_features # Assuming in_features is the full rank for MHA models
    total_rank = num_layers * hidden_size
    k_value = int(args.rank_ratio/2 * total_rank) # Factor of 2 is a legacy setting
        
    top_indices, top_scores, layer_indices_dict = get_top_k_scores(grad_scores_dict, k=k_value)
    
    logging.info(f"Selected top {len(top_indices)} important singular values")
    return top_indices, top_scores, layer_indices_dict

# Add the following functions at the end of the file

def get_top_k_scores(grad_scores_dict, k):
    """
    Get indices and scores of top k important singular values across all layers
    
    Args:
        grad_scores_dict: Dictionary containing gradient importance scores for each layer
        k: Number of important singular values to select
    
    Returns:
        top_indices: List of (layer_index, singular_value_index) tuples for top k important singular values
        top_scores: List of corresponding importance scores
        layer_indices_dict: Dictionary of selected singular value indices for each layer
    """
    # Collect scores from all layers
    all_scores = []
    for layer_idx, scores in grad_scores_dict.items():
        layer_num = int(layer_idx.split('_')[1])
        for i, score in enumerate(scores):
            all_scores.append((layer_num, i, score.item()))
    
    # Sort by score in descending order
    all_scores.sort(key=lambda x: x[2], reverse=True)
    
    # Select top k
    top_k = all_scores[:k]
    
    # Separate indices and scores
    top_indices = [(item[0], item[1]) for item in top_k]
    top_scores = [item[2] for item in top_k]
    
    # Create index dictionary for each layer
    layer_indices_dict = {}
    for layer_idx, singular_idx in top_indices:
        if layer_idx not in layer_indices_dict:
            layer_indices_dict[layer_idx] = []
        layer_indices_dict[layer_idx].append(singular_idx)
    
    return top_indices, top_scores, layer_indices_dict

def visualize_score_distribution(grad_scores_dict, save_path=None, plot_type='boxplot'):
    """
    Visualize the distribution of gradient importance scores
    
    Args:
        grad_scores_dict: Dictionary containing gradient importance scores for each layer
        save_path: Path to save the image
        plot_type: Plot type, 'boxplot' or 'violin'
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Collect scores from all layers
        layer_scores = []
        layer_names = []
        
        for layer_name, scores in grad_scores_dict.items():
            layer_scores.append(scores.cpu().numpy())
            layer_names.append(layer_name)
        
        plt.figure(figsize=(12, 8))
        
        if plot_type == 'boxplot':
            plt.boxplot(layer_scores, labels=layer_names)
            plt.title('Gradient Importance Score Distribution (Box Plot)')
        elif plot_type == 'violin':
            sns.violinplot(data=layer_scores)
            plt.xticks(range(len(layer_names)), layer_names)
            plt.title('Gradient Importance Score Distribution (Violin Plot)')
        
        plt.xlabel('Layer')
        plt.ylabel('Importance Score')
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Distribution plot saved to {save_path}")
        
        plt.close()
        
    except ImportError:
        print("Cannot import matplotlib or seaborn, skipping visualization")
    except Exception as e:
        print(f"Error during visualization: {e}")

def visualize_layer_score_histograms(grad_scores_dict, save_path=None, max_layers=16):
    """
    Draw histograms of importance scores for each layer
    
    Args:
        grad_scores_dict: Dictionary containing gradient importance scores for each layer
        save_path: Path to save the image
        max_layers: Maximum number of layers to display
    """
    try:
        import matplotlib.pyplot as plt
        
        # Limit the number of layers to display
        layer_names = list(grad_scores_dict.keys())[:max_layers]
        n_layers = len(layer_names)
        
        # Calculate subplot layout
        n_cols = min(4, n_layers)
        n_rows = (n_layers + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 3 * n_rows))
        
        for i, layer_name in enumerate(layer_names):
            scores = grad_scores_dict[layer_name].cpu().numpy()
            
            plt.subplot(n_rows, n_cols, i + 1)
            plt.hist(scores, bins=50)
            plt.title(f'{layer_name}')
            plt.xlabel('Importance Score')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Histogram saved to {save_path}")
        
        plt.close()
        
    except ImportError:
        print("Cannot import matplotlib, skipping visualization")
    except Exception as e:
        print(f"Error during visualization: {e}")
