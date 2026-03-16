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

    # --------------------------------------------------------------------------
    # FAST GRAD COMPUTATION VIA VISION TOWER OFFLOADING
    # --------------------------------------------------------------------------
    device = utils.get_dev()
    
    # Enable grad only for Q/K/V up front so gradients are allocated properly when model is moved
    for name, param in model.named_parameters():
        if 'model.layers' in name:
            if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        else:
            param.requires_grad = False

    logging.info("Step 1: Pre-computing input embeddings to allow Vision Tower offloading")
    precomputed_batches = []
    
    # Move only vision components and embeddings to device
    if hasattr(model, 'vision_tower') and model.vision_tower is not None:
        model.vision_tower = model.vision_tower.to(device)
    if hasattr(model, 'multi_modal_projector') and model.multi_modal_projector is not None:
        model.multi_modal_projector = model.multi_modal_projector.to(device)
    model.get_input_embeddings().to(device)

    def move_to_device(obj, target_device):
        if isinstance(obj, torch.Tensor):
            return obj.to(target_device)
        elif isinstance(obj, (list, tuple)):
            return type(obj)(move_to_device(item, target_device) for item in obj)
        elif isinstance(obj, dict):
            return {k: move_to_device(v, target_device) for k, v in obj.items()}
        else:
            return obj

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Pre-computing embeddings"):
            try:
                # 1. Get raw inputs based on tokenizer type
                if tokenizer is None: # SmolVLM
                    inputs, _, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                elif tokenizer == 'hf_v16': # LLaVA Next
                    inputs, _, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                elif 'hf_v16' in str(tokenizer): # handle 'hf_v16_trainfix'
                    inputs, _, _ = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer, label_mode=args.label_mode)
                    output_ids = inputs.get('labels')
                else: # LLaVA
                    input_ids_raw, images, output_ids = gptq_utils.message_to_prompt_train(batch, image_processor, model, tokenizer)
                    inputs = {'input_ids': input_ids_raw}
                    if images is not None:
                        images_tensor, image_sizes = images
                        inputs['pixel_values'] = images_tensor
                        inputs['image_sizes'] = image_sizes
                        
                inputs = move_to_device(inputs, device)
                output_ids = move_to_device(output_ids, device)
                input_ids = inputs.get('input_ids')
                
                # 2. Align token lengths (pad to max_len)
                if input_ids.size(1) != output_ids.size(1):
                    max_len = max(input_ids.size(1), output_ids.size(1))
                    if input_ids.size(1) < max_len:
                        padding = torch.zeros((input_ids.size(0), max_len - input_ids.size(1)), dtype=input_ids.dtype, device=input_ids.device)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                    else:
                        input_ids = input_ids[:, :max_len]
                    if output_ids.size(1) < max_len:
                        padding = torch.full((output_ids.size(0), max_len - output_ids.size(1)), fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
                        output_ids = torch.cat([output_ids, padding], dim=1)
                    else:
                        output_ids = output_ids[:, :max_len]
                        
                inputs['input_ids'] = input_ids
                inputs['attention_mask'] = input_ids.ne(0).to(device)
                
                if hasattr(args, 'token_length') and args.token_length > 0 and 'hf_v16' in str(tokenizer):
                    inputs['input_ids'] = inputs['input_ids'][:,:args.token_length]
                    inputs['attention_mask'] = inputs['attention_mask'][:,:args.token_length]
                    output_ids = output_ids[:,:args.token_length]
                    
                # 3. Compute multimodal embeddings
                inputs_embeds = None
                position_ids = None
                labels = output_ids
                
                if hasattr(model, 'prepare_inputs_labels_for_multimodal'):
                    # LLaVA 1.5 logic
                    (_, position_ids, attention_mask, _, inputs_embeds, labels) = \
                        model.prepare_inputs_labels_for_multimodal(
                            input_ids=inputs['input_ids'],
                            position_ids=None,
                            attention_mask=inputs['attention_mask'],
                            past_key_values=None,
                            labels=output_ids,
                            images=inputs.get('pixel_values'),
                            image_sizes=inputs.get('image_sizes')
                        )
                    inputs['attention_mask'] = attention_mask
                        
                elif "LlavaNext" in type(model).__name__:
                    # LLaVA-Next HF logic
                    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
                    if inputs.get('pixel_values') is not None and inputs['pixel_values'].size(0) > 0:
                        vision_model = model.model if hasattr(model, 'model') else model
                        image_features = vision_model.get_image_features(
                            pixel_values=inputs['pixel_values'],
                            image_sizes=inputs.get('image_sizes'),
                            vision_feature_layer=model.config.vision_feature_layer,
                            vision_feature_select_strategy=model.config.vision_feature_select_strategy
                        )
                        image_features, _ = vision_model.pack_image_features(
                            image_features,
                            inputs.get('image_sizes'),
                            vision_feature_select_strategy=model.config.vision_feature_select_strategy,
                            image_newline=vision_model.image_newline
                        )
                        special_image_mask = (inputs['input_ids'] == model.config.image_token_id).unsqueeze(-1)
                        special_image_mask = special_image_mask.expand_as(inputs_embeds)
                        image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
                        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)
                else:
                    # Generic / Text-only
                    inputs_embeds = model.get_input_embeddings()(inputs['input_ids'])
                    
                precomputed_batches.append({
                    'inputs_embeds': inputs_embeds.cpu(),
                    'attention_mask': inputs['attention_mask'].cpu(),
                    'labels': labels.cpu(),
                    'position_ids': position_ids.cpu() if position_ids is not None else None
                })
                
            except Exception as e:
                import traceback
                logging.warning(f"Failed to pre-compute embeddings for batch: {e}\n{traceback.format_exc()}")
                continue

    if len(precomputed_batches) == 0:
        logging.error("Failed to precompute embeddings for any batch.")
        batch_count = 0
    else:
        logging.info(f"Step 2: Offloading Vision Tower and moving LLM to {device}")
        model = model.to(device)
        if hasattr(model, 'vision_tower') and model.vision_tower is not None:
            model.vision_tower = model.vision_tower.cpu()
        if hasattr(model, 'multi_modal_projector') and model.multi_modal_projector is not None:
            model.multi_modal_projector = model.multi_modal_projector.cpu()
        torch.cuda.empty_cache()
    
        logging.info("Step 3: Calculating gradients using precomputed embeddings")
        model.train()
        batch_count = 0
        accumulation_steps = 1
        
        for batch_data in tqdm(precomputed_batches, desc="Computing Gradients"):
            try:
                inputs_embeds = batch_data['inputs_embeds'].to(device)
                attention_mask = batch_data['attention_mask'].to(device)
                labels = batch_data['labels'].to(device)
                position_ids = batch_data['position_ids'].to(device) if batch_data['position_ids'] is not None else None
                
                with torch.enable_grad():
                    # We can call model() with inputs_embeds, it skips the vision branch natively
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        labels=labels,
                        position_ids=position_ids,
                        input_ids=None,
                        pixel_values=None
                    )
                    loss = outputs[0]
                    loss /= accumulation_steps
                    loss.backward()
                    
                batch_count += 1
                if batch_count % accumulation_steps == 0:
                    for idx, layer in enumerate(model_utils.get_layers(model)):
                        if hasattr(layer.self_attn, 'qkv_svd_info'):
                            svd_info = layer.self_attn.qkv_svd_info
                            q_linear = layer.self_attn.q_proj
                            k_linear = layer.self_attn.k_proj
                            v_linear = layer.self_attn.v_proj
                            
                            if q_linear.weight.grad is not None and k_linear.weight.grad is not None and v_linear.weight.grad is not None:
                                grad_cat = torch.cat([
                                    q_linear.weight.grad.detach().to(torch.float32),
                                    k_linear.weight.grad.detach().to(torch.float32),
                                    v_linear.weight.grad.detach().to(torch.float32)
                                ], dim=0)
                                
                                if args.act_aware:
                                    scaling = svd_info['scaling_matrix_inverse_transpose'].to(device)
                                    if scaling.ndim == 1:
                                        grad_cat = grad_cat * scaling.view(1, -1).to(torch.float32)
                                    elif scaling.ndim == 2:
                                        grad_cat = grad_cat @ scaling.to(torch.float32)
                                        
                                U = svd_info['U'].to(device).to(torch.float32)
                                V = svd_info['V'].to(device).to(torch.float32)
                                S_grad_squared = torch.sum(U * (grad_cat @ V), dim=0).pow(2)
                                
                                if not hasattr(layer.self_attn, 'S_grad_info'):
                                    layer.self_attn.S_grad_info = S_grad_squared
                                else:
                                    layer.self_attn.S_grad_info += S_grad_squared
                                    
                    model.zero_grad()
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logging.error(f"Error during Gradient computation: {e}")
                import traceback
                logging.error(traceback.format_exc())
                model.zero_grad()
                torch.cuda.empty_cache()
                
        # Restore model fully to device for subsequent processing
        logging.info("Restoring vision tower to GPU")
        if hasattr(model, 'vision_tower') and model.vision_tower is not None:
            model.vision_tower = model.vision_tower.to(device)
        if hasattr(model, 'multi_modal_projector') and model.multi_modal_projector is not None:
            model.multi_modal_projector = model.multi_modal_projector.to(device)

    # Normalize S gradient information
    if batch_count == 0:
        logging.error(f"All {len(dataloader)} batches failed during grad computation — "
                      "no S_grad_info was accumulated. Check errors above. "
                      "Cache will NOT be saved to avoid poisoning future runs.")
        return

    logging.info(f'Grad computation succeeded on {batch_count} batch(es)')
    for layer in model_utils.get_layers(model):
        if hasattr(layer.self_attn, 'S_grad_info'):
            layer.self_attn.S_grad_info = layer.self_attn.S_grad_info.div(batch_count//accumulation_steps).sqrt()

    logging.info('finished grad computing')
    # Save S gradient information
    all_grad_info = {}
    for idx, layer in enumerate(model_utils.get_layers(model)):
        if hasattr(layer.self_attn, 'S_grad_info'):
            logging.info(f"Layer {idx}: S_grad_info shape {layer.self_attn.S_grad_info.shape}")
            all_grad_info[f"layer_{idx}"] = layer.self_attn.S_grad_info.cpu()

    if not all_grad_info:
        logging.error("all_grad_info is empty after successful batches — "
                      "S_grad_info was never set (gradients were likely None). "
                      "Cache will NOT be saved.")
        return

    logging.info(f"Saving Grad information cache to {cache_file}...")
    torch.save(all_grad_info, cache_file)
    logging.info("Grad information cache saved successfully!")

# def calib_grad_info(model, dataloader, tokenizer, image_processor, args, use_cache=True, cache_file=None):
#     """
#     Calculate Grad matrix for each layer of the model to evaluate parameter importance
    
#     Args:
#         model: Model to be calibrated
#         tokenizer: Tokenizer
#         image_processor: Image processor
#         args: Parameter configuration
#         use_cache: Whether to use cache
#         cache_file: Cache file path, automatically generated if None
#     """
#     model_id = model.config._name_or_path
    
#     if cache_file is None:
#         cache_dir = "cache"
#         if args.cache_in_log:
#             cache_dir = args.save_path + "/cache"
#         os.makedirs(cache_dir, exist_ok=True)
#         calib_method_info = args.calib_method if hasattr(args, "act_aware") and args.act_aware else "no_act_aware"
#         if args.a_clip_ratio == 1.0:
#             cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma_grad_info.pt")
#         else:
#             cache_file = os.path.join(cache_dir, f"{args.model.replace('/','_')}_aclip{args.a_clip_ratio}_{args.nsamples}_{args.seed}_{calib_method_info}_sigma_grad_info.pt")
#     else:
#         calib_method_info = args.calib_method if hasattr(args, "act_aware") and args.act_aware else "no_act_aware"
#         cache_file = os.path.join(args.cache_file, f"{args.model.replace('/','_')}_{args.nsamples}_{calib_method_info}_sigma_grad_info.pt")

#     # First perform QKV SVD decomposition and store
#     logging.info('start qkv svd for grad')
#     prepare_qkv_svd(model, args)
#     logging.info('finish qkv svd for grad')

#     if os.path.exists(cache_file) and use_cache:
#         # FIX: check cache is non-empty before trusting it
#         cached_data = torch.load(cache_file, map_location="cpu")
#         if len(cached_data) == 0:
#             logging.warning(f"Cache file exists but is empty: {cache_file}")
#             logging.warning("Deleting stale empty cache and recomputing...")
#             os.remove(cache_file)
#         else:
#             logging.info(f"Loading Grad information cache from {cache_file}...")
#             all_grad_info = cached_data
#             for idx, layer in enumerate(model_utils.get_layers(model)):
#                 layer_key = f"layer_{idx}"
#                 if layer_key in all_grad_info:
#                     layer.self_attn.S_grad_info = all_grad_info[layer_key].to(utils.get_dev())
#             logging.info("Successfully loaded Grad information cache!")
#             return

#     print("Starting Grad information calculation...")
#     logging.info('start grad computing')
#     model.eval()

#     device = utils.get_dev()
#     model = model.to(device)

#     accumulation_steps = 1
#     batch_count = 0
#     successful_batches = 0

#     # -------------------------------------------------------------------------
#     # STEP 1: Pre-compute input embeddings (with image tokens merged) for all
#     # batches while the vision tower is still on GPU. After this we can offload
#     # the vision tower and run only the language-model backbone for gradients.
#     # -------------------------------------------------------------------------
#     logging.info("Pre-computing input embeddings for all batches...")
#     precomputed_batches = []

#     def move_to_device(obj, target_device):
#         if isinstance(obj, torch.Tensor):
#             return obj.to(target_device)
#         elif isinstance(obj, (list, tuple)):
#             return type(obj)(move_to_device(item, target_device) for item in obj)
#         elif isinstance(obj, dict):
#             return {k: move_to_device(v, target_device) for k, v in obj.items()}
#         else:
#             return obj

#     with torch.no_grad():
#         for batch in dataloader:
#             try:
#                 if tokenizer is None:  # SmolVLM
#                     inputs, _, output_ids = gptq_utils.message_to_prompt_train(
#                         batch, image_processor, model, tokenizer, label_mode=args.label_mode)
#                 elif tokenizer == 'hf_v16':  # LLaVA-Next
#                     inputs, _, output_ids = gptq_utils.message_to_prompt_train(
#                         batch, image_processor, model, tokenizer, label_mode=args.label_mode)
#                 elif 'hf_v16' in str(tokenizer):  # hf_v16_trainfix
#                     inputs, _, _ = gptq_utils.message_to_prompt_train(
#                         batch, image_processor, model, tokenizer, label_mode=args.label_mode)
#                     output_ids = inputs.get('labels')
#                 else:  # LLaVA
#                     input_ids_raw, images, output_ids = gptq_utils.message_to_prompt_train(
#                         batch, image_processor, model, tokenizer)
#                     inputs = {'input_ids': input_ids_raw}
#                     if images is not None:
#                         images_tensor, image_sizes = images
#                         inputs['pixel_values'] = images_tensor
#                         inputs['image_sizes'] = image_sizes

#                 inputs = move_to_device(inputs, device)
#                 output_ids = move_to_device(output_ids, device)

#                 input_ids = inputs.get('input_ids')

#                 # Align input and label lengths
#                 if input_ids.size(1) != output_ids.size(1):
#                     max_len = max(input_ids.size(1), output_ids.size(1))
#                     if input_ids.size(1) < max_len:
#                         padding = torch.zeros(
#                             (input_ids.size(0), max_len - input_ids.size(1)),
#                             dtype=input_ids.dtype, device=input_ids.device)
#                         input_ids = torch.cat([input_ids, padding], dim=1)
#                     else:
#                         input_ids = input_ids[:, :max_len]
#                     if output_ids.size(1) < max_len:
#                         padding = torch.full(
#                             (output_ids.size(0), max_len - output_ids.size(1)),
#                             fill_value=-100, dtype=output_ids.dtype, device=output_ids.device)
#                         output_ids = torch.cat([output_ids, padding], dim=1)
#                     else:
#                         output_ids = output_ids[:, :max_len]

#                 inputs['input_ids'] = input_ids
#                 inputs['attention_mask'] = input_ids.ne(0).to(device)

#                 if args.token_length > 0 and 'hf_v16' in str(tokenizer):
#                     input_ids = input_ids[:, :args.token_length]
#                     output_ids = output_ids[:, :args.token_length]
#                     inputs['input_ids'] = input_ids
#                     inputs['attention_mask'] = input_ids.ne(0).to(device)

#                 # Try to merge image tokens into embeddings via LLaVA's internal method
#                 if hasattr(model, 'prepare_inputs_labels_for_multimodal'):
#                     (input_ids_prep, position_ids, attention_mask_prep,
#                      past_key_values, inputs_embeds, labels) = \
#                         model.prepare_inputs_labels_for_multimodal(
#                             input_ids=input_ids,
#                             position_ids=None,
#                             attention_mask=inputs['attention_mask'],
#                             past_key_values=None,
#                             labels=output_ids,
#                             pixel_values=inputs.get('pixel_values'),
#                             image_sizes=inputs.get('image_sizes'),
#                         )

#                     if inputs_embeds is not None:
#                         precomputed_batches.append({
#                             'inputs_embeds': inputs_embeds.cpu(),
#                             'attention_mask': (attention_mask_prep.cpu()
#                                                if attention_mask_prep is not None
#                                                else inputs['attention_mask'].cpu()),
#                             'labels': (labels.cpu() if labels is not None
#                                        else output_ids.cpu()),
#                             'position_ids': (position_ids.cpu()
#                                              if position_ids is not None else None),
#                             'use_raw': False,
#                         })
#                     else:
#                         # inputs_embeds is None — fall back to storing raw inputs
#                         precomputed_batches.append({
#                             'inputs': {k: v.cpu() if isinstance(v, torch.Tensor) else v
#                                        for k, v in inputs.items()},
#                             'output_ids': output_ids.cpu(),
#                             'use_raw': True,
#                         })
#                 else:
#                     # Model has no prepare_inputs_labels_for_multimodal — store raw
#                     precomputed_batches.append({
#                         'inputs': {k: v.cpu() if isinstance(v, torch.Tensor) else v
#                                    for k, v in inputs.items()},
#                         'output_ids': output_ids.cpu(),
#                         'use_raw': True,
#                     })

#             except Exception as e:
#                 logging.warning(f"Failed to pre-compute embeddings for batch: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue

#     logging.info(f"Pre-computed embeddings for {len(precomputed_batches)} / {args.nsamples} batches")

#     if len(precomputed_batches) == 0:
#         logging.error("Failed to pre-compute any batch embeddings. Aborting grad computation.")
#         return

#     # -------------------------------------------------------------------------
#     # STEP 2: Offload vision tower + projector — no longer needed for grad pass
#     # -------------------------------------------------------------------------
#     logging.info("Offloading vision tower to CPU to free memory for grad computation...")
#     vision_tower = None
#     vision_tower_device = None
#     if hasattr(model, 'vision_tower') and model.vision_tower is not None:
#         vision_tower = model.vision_tower
#         vision_tower_device = next(model.vision_tower.parameters()).device
#         model.vision_tower = model.vision_tower.cpu()
#         torch.cuda.empty_cache()
#         logging.info("Vision tower offloaded to CPU")

#     if hasattr(model, 'multi_modal_projector') and model.multi_modal_projector is not None:
#         model.multi_modal_projector = model.multi_modal_projector.cpu()
#         torch.cuda.empty_cache()
#         logging.info("Multi-modal projector offloaded to CPU")

#     # -------------------------------------------------------------------------
#     # STEP 3: Enable gradients only on Q/K/V of the language-model layers
#     # -------------------------------------------------------------------------
#     model.train()
#     for name, param in model.named_parameters():
#         if 'model.layers' in name:
#             if 'q_proj' in name or 'k_proj' in name or 'v_proj' in name:
#                 param.requires_grad = True
#             else:
#                 param.requires_grad = False
#         else:
#             param.requires_grad = False

#     # -------------------------------------------------------------------------
#     # STEP 4: Gradient computation loop over pre-computed batches
#     # -------------------------------------------------------------------------
#     for batch_data in tqdm(precomputed_batches, desc="Computing Gradient Information"):
#         torch.cuda.empty_cache()
#         try:
#             if batch_data.get('use_raw'):
#                 # Vision tower is offloaded — cannot process raw pixel_values
#                 logging.warning("Skipping raw batch (vision tower is offloaded, no inputs_embeds available)")
#                 continue

#             inputs_embeds = batch_data['inputs_embeds'].to(device)
#             attention_mask = batch_data['attention_mask'].to(device)
#             labels = batch_data['labels'].to(device)
#             position_ids = (batch_data['position_ids'].to(device)
#                             if batch_data['position_ids'] is not None else None)

#             with torch.enable_grad():
#                 outputs = model.language_model(
#                     inputs_embeds=inputs_embeds,
#                     attention_mask=attention_mask,
#                     labels=labels,
#                     position_ids=position_ids,
#                 )
#                 loss = outputs[0]
#                 loss = loss / accumulation_steps
#                 loss.backward()

#             batch_count += 1

#             if batch_count % accumulation_steps == 0:
#                 for idx, layer in enumerate(model_utils.get_layers(model)):
#                     if hasattr(layer.self_attn, 'qkv_svd_info'):
#                         svd_info = layer.self_attn.qkv_svd_info
#                         q_linear = layer.self_attn.q_proj
#                         k_linear = layer.self_attn.k_proj
#                         v_linear = layer.self_attn.v_proj

#                         if (q_linear.weight.grad is not None and
#                                 k_linear.weight.grad is not None and
#                                 v_linear.weight.grad is not None):
#                             grad_cat = torch.cat([
#                                 q_linear.weight.grad.detach().to(torch.float32),
#                                 k_linear.weight.grad.detach().to(torch.float32),
#                                 v_linear.weight.grad.detach().to(torch.float32)
#                             ], dim=0).to(device)

#                             if args.act_aware:
#                                 scaling_matrix_inverse_transpose = \
#                                     svd_info['scaling_matrix_inverse_transpose'].to(device)
#                                 if scaling_matrix_inverse_transpose.ndim == 1:
#                                     grad_cat = grad_cat * \
#                                         scaling_matrix_inverse_transpose.view(1, -1).to(torch.float32)
#                                 elif scaling_matrix_inverse_transpose.ndim == 2:
#                                     grad_cat = grad_cat @ \
#                                         scaling_matrix_inverse_transpose.to(torch.float32)

#                             U = svd_info['U'].to(device).to(torch.float32)
#                             V = svd_info['V'].to(device).to(torch.float32)
#                             S_grad = torch.sum(U * (grad_cat @ V), dim=0)
#                             S_grad_squared = S_grad.pow(2)

#                             if not hasattr(layer.self_attn, 'S_grad_info'):
#                                 layer.self_attn.S_grad_info = S_grad_squared
#                             else:
#                                 layer.self_attn.S_grad_info += S_grad_squared

#                 model.zero_grad()
#                 successful_batches += 1

#         except Exception as e:
#             print(f"Error occurred during Grad information calculation: {e}")
#             import traceback
#             print("Detailed error information:")
#             traceback.print_exc()
#             model.zero_grad()
#             torch.cuda.empty_cache()
#             continue

#     # -------------------------------------------------------------------------
#     # STEP 5: Restore vision tower + projector to GPU
#     # -------------------------------------------------------------------------
#     if vision_tower is not None:
#         logging.info("Restoring vision tower to GPU...")
#         model.vision_tower = vision_tower.to(vision_tower_device)
#         if hasattr(model, 'multi_modal_projector'):
#             model.multi_modal_projector = model.multi_modal_projector.to(vision_tower_device)
#         torch.cuda.empty_cache()
#         logging.info("Vision tower restored to GPU")

#     # -------------------------------------------------------------------------
#     # STEP 6: Abort if nothing succeeded — never save an empty cache
#     # -------------------------------------------------------------------------
#     if successful_batches == 0:
#         logging.error(
#             f"All batches failed during grad computation — "
#             "S_grad_info was not set on any layer. "
#             "Cache will NOT be saved to avoid poisoning future runs. "
#             "Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
#             "or freeing GPU memory before calling calib_grad_info."
#         )
#         return

#     logging.info(f"Grad computation succeeded on {successful_batches}/{batch_count} batches")

#     # -------------------------------------------------------------------------
#     # STEP 7: Normalize using successful batch count only
#     # -------------------------------------------------------------------------
#     for layer in model_utils.get_layers(model):
#         if hasattr(layer.self_attn, 'S_grad_info'):
#             layer.self_attn.S_grad_info = \
#                 layer.self_attn.S_grad_info.div(successful_batches).sqrt()

#     logging.info('finished grad computing')

#     # -------------------------------------------------------------------------
#     # STEP 8: Collect and save — guard against empty dict
#     # -------------------------------------------------------------------------
#     all_grad_info = {}
#     for idx, layer in enumerate(model_utils.get_layers(model)):
#         if hasattr(layer.self_attn, 'S_grad_info'):
#             print(f"Layer {idx}: {layer.self_attn.S_grad_info.shape}")
#             all_grad_info[f"layer_{idx}"] = layer.self_attn.S_grad_info.cpu()

#     if not all_grad_info:
#         logging.error(
#             "all_grad_info is empty after successful batch processing — "
#             "S_grad_info was not set on any layer. "
#             "Check that qkv_svd_info is present and gradients are non-None. "
#             "Cache will NOT be saved."
#         )
#         return

#     logging.info(f"Saving Grad information cache to {cache_file}...")
#     torch.save(all_grad_info, cache_file)
#     logging.info("Grad information cache saved successfully!")



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
    
    # DEBUG
    total_entries = sum(v.numel() for v in grad_scores_dict.values())
    nonzero = sum((v != 0).sum().item() for v in grad_scores_dict.values())
    max_score = max(v.max().item() for v in grad_scores_dict.values()) if grad_scores_dict else 0
    logging.info(f"Score cache: {len(grad_scores_dict)} layers, {total_entries} total, {nonzero} nonzero, max={max_score:.6f}, k={k_value}")
    #end of debug

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
