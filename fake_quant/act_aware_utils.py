import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import utils
import gptq_utils
import data_utils
import model_utils
import logging


@torch.no_grad()
def calib_input_distribution(model, dataloader, tokenizer, image_processor, args, method, use_cache=True, cache_file=None):
    model_id = model.config._name_or_path
    
    # Use the passed method parameter, if None try to get from args
    if method is None and hasattr(args, "calib_method"):
        method = args.calib_method
    
    if cache_file is None:
        cache_dir = "cache"
        
        if args.cache_in_log:
            cache_dir = args.save_path + "/cache"
        
        # cache_dir = "cacheseed"
        os.makedirs(cache_dir, exist_ok=True)
        # Add rotate information to cache file name
        rotate_info = "rotated" if hasattr(args, "rotate") and args.rotate else "norotate"
        cache_file = os.path.join(cache_dir, f"{model_id.replace('/','_')}_{args.nsamples}_{args.seed}_calib_input_distribution_{method}.pt")
    else:
        cache_file = os.path.join(args.cache_file, f"{model_id.replace('/','_')}_{args.nsamples}_calib_input_distribution_{method}.pt")

    if os.path.exists(cache_file) and use_cache:
        logging.info(f"Loading calibration cache from {cache_file}...")
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        #### todo: add cholesky load
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'lm_head' not in name and 'model.layers' in name:
                if "cholesky" in method:
                    module.scaling_diag_matrixS = all_scaling_diag_matrix[name].float().to(
                        module.weight.device
                    ) if all_scaling_diag_matrix[name] is not None else None
                else:
                    try:
                        module.scaling_diag_matrix = all_scaling_diag_matrix[name].float().to(
                            module.weight.device
                        ) if all_scaling_diag_matrix[name] is not None else None
                    except:
                        # add support for lower transformer version cache loading.
                        name = name.replace("model.language_model", "language_model.model")
                        module.scaling_diag_matrix = all_scaling_diag_matrix[name].float().to(
                            module.weight.device
                        ) if all_scaling_diag_matrix[name] is not None else None
        logging.info("Successfully loaded calibration cache!")
        return
    
    logging.info("Starting online calibration...")
    logging.info(cache_file)
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean.float()
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            ).float()
        elif "cholesky" in method:
            inp = input[0].detach().float()
            if inp.dim() == 2:   # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.raw_scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum
            torch.cuda.empty_cache()
        # abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
        # module.scaling_diag_matrix += abs_max
    if "cholesky" in method:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'model.layers' in name and 'k_proj' in name:
                module.raw_scaling_diag_matrix = 0
                module.register_forward_hook(hook)
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'model.layers' in name:
                module.scaling_diag_matrix = 0
                module.register_forward_hook(hook)

    # get activation distribution
    class Catcherout(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inp = self.module(inp)
            # cache['i'] += 1
            raise ValueError

    layers = model_utils.get_layers(model)
    num_layers = len(layers)
    layer = layers[num_layers-1]
    layer.mlp = Catcherout(layer.mlp)
    
    # Ensure the entire model is on CUDA
    model = model.to(utils.get_dev())

    total_seq_length = 0
    
    for batch in dataloader:
            try:
                # Ensure model and input are on the same device
                device = utils.get_dev()  # Get the device where the model is currently located
                
                ## TODO:add model type as criteria
                # Use model to generate output
                if tokenizer is None or 'hf_v16' in str(tokenizer): # SmolVLM/llava 1.6
                    # Use message_to_prompt to process batch data
                    inputs, _ = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
                    inputs = inputs.to(device)
                    
                    # Assume each batch contains only one sample, directly take its sequence length
                    seq_length = inputs['input_ids'].shape[1]
                    total_seq_length += seq_length
                    
                    model.generate(**inputs,
                                    max_new_tokens=1,
                                    use_cache=True,)
                else: # LLaVA
                    # Use message_to_prompt to process batch data
                    input_ids, images = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
                    
                    # Ensure input_ids and images are on the correct device
                    input_ids = input_ids.to(device)
                    image_sizes = None
                    if images is not None:
                        images, image_sizes = images
                        images = images.to(device)
                        
                    # Assume each batch contains only one sample, directly take its sequence length
                    seq_length = input_ids.shape[1]
                    total_seq_length += seq_length
        
                    from llava.constants import IMAGE_TOKEN_INDEX
                    # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
                    # print(num_images)
                    #breakpoint()
                
                    model.generate(input_ids, images=images,
                                   image_sizes = image_sizes,
                                    do_sample=False,
                                    max_new_tokens=1,
                                    use_cache=True,)
            except ValueError:
                # print(f"Error during calibration: {e}")
                # print("finish prefill exit")
                # return
                pass
                # break
    # for batch in tqdm(calib_loader):
    #     # print(batch)
    #     batch = {k: v.to(model.device) for k, v in batch.items()}
    #     model(**batch)

    logging.info("Random Seed: %s", args.seed)
    logging.info("Total sequence length: %s", total_seq_length)
    
    layer.mlp = layer.mlp.module

    if "cholesky" in method:
        # remove and compute cholesky decomposition
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'model.layers' in name and 'k_proj' in name:
                module._forward_hooks.clear()
                module.raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.cpu()
        # profiling_mat = {}
        layers = model_utils.get_layers(model) # skip vision for now
        logging.info("Start Cholesky Decomposition...")
        for i in tqdm(range(len(layers))):
            # layer_profile = {}
            for name, module in layers[i].named_modules():
                if isinstance(module, nn.Linear) and 'k_proj' in name:
                    raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.double().to(utils.get_dev())
                    try:
                        scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                    except Exception as e:
                        print("Warning: eigen scaling_diag_matrix is not positive!")
                        eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                        raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(utils.get_dev())
                        scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                        eigenvalues = None
                        del eigenvalues
                    # layer_profile[name] = scaling_diag_matrix.cpu()
                    layers[i].self_attn.q_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                    layers[i].self_attn.k_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                    layers[i].self_attn.v_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                    scaling_diag_matrix = raw_scaling_diag_matrix = module.raw_scaling_diag_matrix = None
                    del scaling_diag_matrix, raw_scaling_diag_matrix, module.raw_scaling_diag_matrix
                    torch.cuda.empty_cache()
            # profiling_mat[i] = layer_profile

        all_scaling_diag_matrix = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'model.layers' in name:
                try:
                    all_scaling_diag_matrix[name] = module.scaling_diag_matrixS
                except:
                    all_scaling_diag_matrix[name] = None
    else:
        # remove and save scaling_diag_matrix
        all_scaling_diag_matrix = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'model.layers' in name:
                module._forward_hooks.clear()
                all_scaling_diag_matrix[name] = module.scaling_diag_matrix
            
    logging.info(f"Saving calibration cache to {cache_file}...")
    torch.save(all_scaling_diag_matrix, cache_file)
    logging.info("Calibration cache saved successfully!")






@torch.no_grad()
def calib_input_distributionlowresources(model, tokenizer, image_processor, args, method, use_cache=True, cache_file=None, dataloader=None):
    model_id = model.config._name_or_path
    # use_cache = model.config.use_cache
    # model.config.use_cache = False
    # Use the passed method parameter, if None try to get from args
    if method is None and hasattr(args, "calib_method"):
        method = args.calib_method
    
    if cache_file is None:
        cache_dir = "cache"
        if args.cache_in_log:
            cache_dir = args.save_path + "/cache"
        # cache_dir = args.save_path + "/cache"
        # cache_dir = "cacheseed"
        os.makedirs(cache_dir, exist_ok=True)
        # Add rotate information to cache file name
        rotate_info = "rotated" if hasattr(args, "rotate") and args.rotate else "norotate"
        cache_file = os.path.join(cache_dir, f"{model_id.replace('/','_')}_{args.nsamples}_{args.seed}_calib_input_distribution_{method}.pt")
    else:
        cache_file = os.path.join(args.cache_file, f"{model_id.replace('/','_')}_{args.nsamples}_calib_input_distribution_{method}.pt")

    if os.path.exists(cache_file) and use_cache:
        logging.info(f"Loading calibration cache from {cache_file}...")
        all_scaling_diag_matrix = torch.load(cache_file, map_location="cpu")
        #### todo: add cholesky load
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'lm_head' not in name and 'model.layers' in name:
                module.scaling_diag_matrixS = all_scaling_diag_matrix[name].to(
                    module.weight.device
                ) if all_scaling_diag_matrix[name] is not None else None
        logging.info("Successfully loaded calibration cache!")
        return
    
    logging.info("Starting online calibration...")
    model.eval()
    # set hook for every Linear layers

    def hook(module, input, output):
        if "abs_mean" in method:
            abs_mean = input[0].abs().mean(dim=-2).detach().view(-1)
            module.scaling_diag_matrix += abs_mean
        elif "abs_max" in method:
            abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
            module.scaling_diag_matrix = torch.where(
                abs_max > module.scaling_diag_matrix,
                abs_max,
                module.scaling_diag_matrix,
            )
        elif "cholesky" in method:
            inp = input[0].detach().float()
            if inp.dim() == 2:   # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.raw_scaling_diag_matrix += adds_sum
            del inp, adds, adds_sum
            torch.cuda.empty_cache()
        # abs_max = input[0].abs().amax(dim=-2).detach().view(-1)
        # module.scaling_diag_matrix += abs_max
    # if "cholesky" in method:
    #     for name, module in model.named_modules():
    #         if isinstance(module, nn.Linear) and 'model.layers' in name:
    #             module.raw_scaling_diag_matrix = 0
    #             module.register_forward_hook(hook)
    # else:
    #     for name, module in model.named_modules():
    #         if isinstance(module, nn.Linear):
    #             module.scaling_diag_matrix = 0
    #             module.register_forward_hook(hook)

    # get activation distribution
    if dataloader is None:
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
        dataloader, _  = calib_loader
    inps = []
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings':[]}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            # inps[cache['i']] = inp
            inps.append(inp[0])
            cache['i'] += 1
            cache['attention_mask'].append(kwargs['attention_mask'])
            cache['position_embeddings'].append(kwargs['position_embeddings'])
            raise ValueError
    layers = model_utils.get_layers(model)
    layers[0] = Catcher(layers[0])
    
    # Ensure the entire model is on CUDA
    model = model.to(utils.get_dev())
    for batch in dataloader:
            try:
                # Ensure model and input are on the same device
                device = utils.get_dev()  # Get the device where the model is currently located
                if tokenizer is None or 'hf_v16' in str(tokenizer): # SmolVLM/llava1.6
                    # Use message_to_prompt to process batch data
                    inputs, _ = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
                    inputs = inputs.to(device)
                    
                    # Assume each batch contains only one sample, directly take its sequence length
                    # seq_length = inputs['input_ids'].shape[1]
                    # total_seq_length += seq_length
                    
                    model.generate(**inputs,
                                    max_new_tokens=1,
                                    use_cache=True,)
                else: # LLaVA
                    # Use message_to_prompt to process batch data
                    input_ids, images = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
                    
                    # Ensure input_ids and images are on the correct device
                    input_ids = input_ids.to(device)
                    image_sizes = None
                    if images is not None:
                        images, image_sizes = images
                        images = images.to(device)
                        
                    # Assume each batch contains only one sample, directly take its sequence length
                    # seq_length = input_ids.shape[1]
                    # total_seq_length += seq_length
        
                    from llava.constants import IMAGE_TOKEN_INDEX
                    # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
                    # print(num_images)
                    #breakpoint()
                
                    model.generate(input_ids, images=images,
                                   image_sizes = image_sizes,
                                    do_sample=False,
                                    max_new_tokens=1,
                                    use_cache=True,)
            except ValueError:
                # print(f"Error during calibration: {e}")
                # print("finish prefill exit")
                # return
                pass
                # break
    # for batch in tqdm(calib_loader):
    #     # print(batch)
    #     batch = {k: v.to(model.device) for k, v in batch.items()}
    #     model(**batch)
    # model.config.use_cache = use_cache
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    # TODO: add model type to decide names to offload emb/norm
    # try:
    #     model.model.embed_tokens = model.model.embed_tokens.cpu()
    #     model.model.norm = model.model.norm.cpu()
    # except:
    #     model.model.text_model.embed_tokens = model.model.text_model.embed_tokens.cpu()
    #     model.model.text_model.norm = model.model.text_model.norm.cpu()
    torch.cuda.empty_cache()
    outs = [None] * len(inps)
    for i in range(len(layers)):
        layer = layers[i].to(utils.get_dev())
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and 'k_proj' in name: #qkv share input
                module.raw_scaling_diag_matrix = 0
                module.register_forward_hook(hook)

        for j in range(len(inps)):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=cache['attention_mask'][j], position_embeddings=cache['position_embeddings'][j])[0][0]

        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear) and 'k_proj' in name:
                module._forward_hooks.clear()
                module.raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.cpu()
        inps = outs
    logging.info("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        # layer_profile = {}
        for name, module in layers[i].named_modules():
            if isinstance(module, nn.Linear) and 'k_proj' in name:
                raw_scaling_diag_matrix = module.raw_scaling_diag_matrix.double().to(utils.get_dev())
                try:
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                except Exception as e:
                    print("Warning: eigen scaling_diag_matrix is not positive!")
                    eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                    raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(utils.get_dev())
                    scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix).float()
                    eigenvalues = None
                    del eigenvalues
                # layer_profile[name] = scaling_diag_matrix.cpu()
                layers[i].self_attn.q_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                layers[i].self_attn.k_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                layers[i].self_attn.v_proj.scaling_diag_matrixS = scaling_diag_matrix.cpu()
                scaling_diag_matrix = raw_scaling_diag_matrix = module.raw_scaling_diag_matrix = None
                del scaling_diag_matrix, raw_scaling_diag_matrix, module.raw_scaling_diag_matrix
                torch.cuda.empty_cache()
        # profiling_mat[i] = layer_profile

    all_scaling_diag_matrix = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'model.layers' in name:
            try:
                all_scaling_diag_matrix[name] = module.scaling_diag_matrixS
            except:
                all_scaling_diag_matrix[name] = None
            
    logging.info(f"Saving calibration cache to {cache_file}...")
    torch.save(all_scaling_diag_matrix, cache_file)
    logging.info("Calibration cache saved successfully!")
