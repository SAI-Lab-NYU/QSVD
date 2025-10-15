import torch
import typing
import transformers
import utils
import os
import logging
import llava
import quant_utils

OPT_MODEL = transformers.models.opt.modeling_opt.OPTForCausalLM
OPT_LAYER = transformers.models.opt.modeling_opt.OPTDecoderLayer
LLAMA_MODEL = transformers.models.llama.modeling_llama.LlamaForCausalLM
LLAMA_LAYER = transformers.models.llama.modeling_llama.LlamaDecoderLayer
# LLAVA_MODEL = llava.model.language_model.llava_llama.LlavaLlamaForCausalLM
LLAVA_MODEL = llava.model.LlavaLlamaForCausalLM
LLAVA_NEXT_HF = transformers.models.llava_next.modeling_llava_next.LlavaNextForConditionalGeneration
LLAMAV_MODLE = transformers.models.mllama.modeling_mllama.MllamaForConditionalGeneration
SMOVLM_MODEL = transformers.models.idefics3.modeling_idefics3.Idefics3ForConditionalGeneration
# CLIP_MODLE = llava.model.CLIP 
# from transformers.models.mllama.modeling_mllama import MllamaForConditionalGeneration

def model_type_extractor(model):
    if isinstance(model, LLAMA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, OPT_MODEL):
        return OPT_MODEL
    elif isinstance(model, LLAVA_MODEL):
        return LLAMA_MODEL
    elif isinstance(model, LLAVA_NEXT_HF):
        return LLAVA_NEXT_HF
    elif isinstance(model, LLAMAV_MODLE):
        return LLAMAV_MODLE
    elif isinstance(model, SMOVLM_MODEL):
        return SMOVLM_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')

def skip(*args, **kwargs):
    # This is a helper function to save time during the initialization! 
    pass

def get_rope_function_name(model):
    if isinstance(model, LLAMA_MODEL):
        return "apply_rotary_pos_emb"
    if isinstance(model, SMOVLM_MODEL):
        return "apply_rotary_pos_emb"
    if isinstance(model, LLAVA_NEXT_HF):
        return "apply_rotary_pos_emb"
    raise NotImplementedError


def get_layers(model):
    if type(model) == OPT_MODEL:
        return model.model.decoder.layers
    if type(model) == LLAMA_MODEL:
        return model.model.layers
    if type(model) == LLAVA_MODEL:
        return model.model.layers
    if type(model) ==LLAVA_NEXT_HF:
        return model.language_model.layers
    if type(model) == SMOVLM_MODEL:
        return model.model.text_model.layers
    raise NotImplementedError

def get_llava(model_name, hf_token=None):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if 'llava-hf' in model_name.lower():
        import transformers
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
            AutoProcessor,
            LlavaForConditionalGeneration,
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(
                    model_name, 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True,
                    device_map='cpu',
                )
        model.seqlen=2048
        processor = LlavaNextProcessor.from_pretrained(model_name)
        if hf_token =='train_fix':
            return model, 'hf_v16_train_fix', processor
        return model, 'hf_v16', processor
        
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    tokenizer, model, image_processor, seqlen = load_pretrained_model(
        model_path=model_name,
        model_base=None,
        model_name=get_model_name_from_path(model_name),
        device="cpu",
        # attn_implementation='sdpa',# to ensure catcher catch attention_mask, Look into this
        # **{"use_cache": False},
    )
    # model.eval()
    model.seqlen = seqlen
    return model, tokenizer, image_processor
    


def get_llama(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.LlamaForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                          use_auth_token=hf_token,
                                                          low_cpu_mem_usage=True)
    model.seqlen = 2048
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model



def get_opt(model_name):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = transformers.OPTForCausalLM.from_pretrained(model_name, torch_dtype='auto',
                                                        low_cpu_mem_usage=True)
    model.seqlen = model.config.max_position_embeddings
    logging.info('---> Loading {} Model with seq_len: {}'.format(model_name, model.seqlen))
    return model


def get_model(
    model_name, hf_token=None
):
    if 'llama' in model_name and 'vision' not in model_name:
        return get_llama(model_name, hf_token)
    elif 'opt' in model_name:
        return get_opt(model_name)
    elif 'llava' in model_name:
        return get_llava(model_name, hf_token)
    elif 'llama' in model_name and 'vision'  in model_name:
        return get_llamav(model_name, hf_token)
    elif 'SmolVLM' in model_name:
        return get_smovlm(model_name, hf_token)
    else:
        raise ValueError(f'Unknown model {model_name}')


def get_smovlm(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration
    import os.path as osp
    def splitlen(s, sym='/'):
        return len(s.split(sym))
    assert osp.exists(model_name) or splitlen(model_name) == 2
    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )
    model.seqlen = 2048
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def get_llamav(model_name, hf_token):
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import MllamaForConditionalGeneration, MllamaProcessor
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        device_map='cpu',
    )
    processor = MllamaProcessor.from_pretrained(model_name, use_safetensors=True)
    return model, processor


def get_model_type(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif isinstance(model, LLAMA_MODEL):
        model_type = LLAMA_MODEL
    elif isinstance(model, LLAVA_MODEL):
        model_type = LLAMA_MODEL # for Llava LM usage only, without changing code much
    elif isinstance(model, LLAVA_NEXT_HF):
        model_type = LLAVA_NEXT_HF
    elif isinstance(model, LLAMAV_MODLE):
        model_type = LLAMAV_MODLE
    elif isinstance(model, SMOVLM_MODEL):
        return SMOVLM_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type


def get_model_typevit(model):
    if isinstance(model, OPT_MODEL):
        model_type = OPT_MODEL
    elif type(model) == LLAVA_MODEL:
        model_type = LLAVA_MODEL
    elif type(model) == LLAVA_NEXT_HF:
        model_type = LLAVA_MODEL # to have vit different from llava? # will this cause projector problem?
    elif type(model) == LLAMAV_MODLE:
        model_type = LLAMAV_MODLE
    elif type(model) == SMOVLM_MODEL:
        model_type =  SMOVLM_MODEL
    else:
        raise ValueError(f'Unknown model type {model}')
    return model_type

def get_embeddings(model, model_type) -> list[torch.nn.Module]:
    if model_type == LLAMA_MODEL:
        return [model.model.embed_tokens]
    elif model_type == OPT_MODEL:
        return [model.model.decoder.embed_tokens, model.model.decoder.embed_positions]
    elif model_type == SMOVLM_MODEL:
        return [model.model.text_model.embed_tokens]
    elif model_type == LLAVA_NEXT_HF:
        return [model.language_model.embed_tokens]
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_projector(model, model_type) -> list[torch.nn.Module]:
    if model_type == LLAVA_MODEL or model_type == LLAMA_MODEL:
        return [model.model.mm_projector]
    elif model_type == LLAMAV_MODLE:
        return [model.multi_modal_projector]
    elif model_type == LLAVA_NEXT_HF:
        return [model.multi_modal_projector]
    elif model_type == SMOVLM_MODEL:
        return [model.model.connector.modality_projection.proj] # no bias
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_transformer_layers(model, model_type):
    if model_type == LLAMA_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == OPT_MODEL:
        return [layer for layer in model.model.decoder.layers]
    elif model_type == LLAVA_MODEL:
        return [layer for layer in model.model.layers]
    elif model_type == SMOVLM_MODEL:
        return [layer for layer in model.model.text_model.layers]
    elif model_type == LLAVA_NEXT_HF:
        return [layer for layer in model.language_model.layers]
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_vit_layers(model, model_type):
    if type(model) == LLAVA_NEXT_HF: # to surpass vit model type in llava_model for attn/mlp rotation simplicity
        return [layer for layer in model.vision_tower.vision_model.encoder.layers] # do not know if this works
    elif model_type == LLAVA_MODEL:
        return [layer for layer in model.model.vision_tower.vision_tower.vision_model.encoder.layers]
    elif model_type == LLAMAV_MODLE:
        return [layer for layer in model.vision_model.transformer.layers] + [layer for layer in model.vision_model.global_transformer.layers]
    elif model_type == SMOVLM_MODEL:
        return [layer for layer in model.model.vision_model.encoder.layers]
    
    else:
        raise ValueError(f'Unknown model type {model_type}')


def get_vit_config(model):
    if type(model) == LLAVA_MODEL:
        return model.model.vision_tower.config
    elif type(model) == LLAVA_NEXT_HF:
        return model.vision_tower.config
    else:
        raise ValueError(f'Unknown model type {type(model)}')

def get_lm_config(model):
    if type(model) == LLAVA_NEXT_HF:
        return model.config.text_config
    else:
        return model.config
    

def get_lm_head(model, model_type):
    if model_type in [LLAMA_MODEL, SMOVLM_MODEL]:
        return model.lm_head
    elif model_type == OPT_MODEL:
        return model.lm_head
    elif model_type == LLAVA_NEXT_HF:
        return model.lm_head
    else:
        raise ValueError(f'Unknown model type {model_type}')

def get_pre_head_layernorm(model, model_type):
    if model_type in [LLAMA_MODEL]:
        pre_head_layernorm = model.model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type in [SMOVLM_MODEL]:
        pre_head_layernorm = model.model.text_model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.llama.modeling_llama.LlamaRMSNorm)
    elif model_type == OPT_MODEL:
        pre_head_layernorm = model.model.decoder.final_layer_norm
        assert pre_head_layernorm is not None
    elif model_type == LLAVA_NEXT_HF:
        pre_head_layernorm = model.language_model.norm
        assert isinstance(pre_head_layernorm,
                          transformers.models.llama.modeling_llama.LlamaRMSNorm)
    else:
        raise ValueError(f'Unknown model type {model_type}')
    return pre_head_layernorm

def get_mlp_bottleneck_size(model):
    model_type = get_model_type(model)
    if model_type == LLAMA_MODEL:
        return model.config.intermediate_size
    elif model_type == OPT_MODEL:
        return model.config.ffn_dim
    else:
        raise ValueError(f'Unknown model type {model_type}')

def replace_modules(
    root: torch.nn.Module,
    type_to_replace,
    new_module_factory,
    replace_layers: bool,
) -> None:
    """Replace modules of given type using the supplied module factory.

    Perform a depth-first search of a module hierarchy starting at root
    and replace all instances of type_to_replace with modules created by
    new_module_factory. Children of replaced modules are not processed.

    Args:
        root: the root of the module hierarchy where modules should be replaced
        type_to_replace: a type instances of which will be replaced
        new_module_factory: a function that given a module that should be replaced
            produces a module to replace it with.
    """
    for name, module in root.named_children():
        new_module = None
        if isinstance(module, type_to_replace):
            if replace_layers:  # layernorm_fusion.replace_layers case where transformer layers are replaced
                new_module = new_module_factory(module, int(name))
            else:  # layernorm_fusion.fuse_modules case where layernorms are fused
                new_module = new_module_factory(module)
        elif len(list(module.children())) > 0:
            replace_modules(module, type_to_replace, new_module_factory, replace_layers)

        if new_module is not None:
            setattr(root, name, new_module)


class RMSN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).sum(-1, keepdim=True) / self.mean_dim
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


class RMSLN(torch.nn.Module):
    """
    This class implements the Root Mean Square Normalization (RMSN) layer.
    We use the implementation from LLAMARMSNorm here:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L75
    """

    def __init__(self, mean_dim: int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.mean_dim = mean_dim
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        if x.dtype == torch.float16:
            x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x.to(input_dtype)


class LN_(torch.nn.Module):
    def __init__(self, ori_layer_norm) -> None:
        super().__init__()
        self.ln = ori_layer_norm


    def forward(self, x):
        out = self.ln(x)
        outdtype = out.dtype
        out_ = out.double()
        out_ = out_ - out_.mean(dim=-1, keepdim=True)
        return out_.to(outdtype)


class LNRotWrapper(torch.nn.Module):
    def __init__(self, module:None):
        super().__init__()
        assert isinstance(module, LN_) or isinstance(module, torch.nn.LayerNorm)
        self.module = module
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.online_random_had = False
        self.had_dim = 0
        self.fp32_had = False
    
    def forward(self, x):
        x_dtype = x.dtype
        x = self.module(x)
        if self.online_random_had:
            if self.fp32_had:
                x = (x.double()@self.had_K).to(x_dtype)
            else:
                x = x@self.had_K.to(x_dtype)
        # Rotate, if needed
        if self.online_full_had:
            import hadamard_utils
            if self.fp32_had: # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else: # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
        return x.to(x_dtype)

def get_layer_io_save_path(args):
    return os.path.join(args.save_path, 'layer_io', f'{args.layer_idx:03d}.pt')

def capture_layer_io(model_type, layer, layer_input):
    def hook_factory(module_name, captured_vals, is_input):
        def hook(module, input, output):
            if is_input:
                captured_vals[module_name].append(input[0].detach().cpu())
            else:
                captured_vals[module_name].append(output.detach().cpu())
        return hook

    handles = []

    if model_type == LLAMA_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'o_proj': [],
            'gate_proj': [],  # up_proj has the same input as gate_proj
            'down_proj': []
        }

        captured_outputs = {
            'v_proj': [],
        }

        for name in captured_inputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            module = getattr(layer.self_attn, name, None) or getattr(layer.mlp, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))

    elif model_type == OPT_MODEL:
        captured_inputs = {
            'k_proj': [],  # q_proj, v_proj has the same input as k_proj
            'out_proj': [],
            'fc1': [],
            'fc2': []
        }
        captured_outputs = {
            'v_proj': [],
        }
        for name in captured_inputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_inputs, True)))

        for name in captured_outputs.keys():
            # In OPT, fc1 and fc2 are directly contained in OPTDecoderLayer
            module = getattr(layer.self_attn, name, None) or getattr(layer, name, None)
            handles.append(module.register_forward_hook(hook_factory(name, captured_outputs, False)))
    else:
        raise ValueError(f'Unknown model type {model_type}')

    # Process each sequence in the batch one by one to avoid OOM.
    for seq_idx in range(layer_input.shape[0]):
        # Extract the current sequence across all dimensions.
        seq = layer_input[seq_idx:seq_idx + 1].to(utils.DEV)
        # Perform a forward pass for the current sequence.
        layer(seq)

    # After processing all sequences, concatenate the accumulated inputs for each sub-layer across the batch.
    for module_name in captured_inputs:
        captured_inputs[module_name] = torch.cat(captured_inputs[module_name], dim=0)
    for module_name in captured_outputs:
        captured_outputs[module_name] = torch.cat(captured_outputs[module_name], dim=0)

    # Cleanup.
    for h in handles:
        h.remove()

    return {
        'input': captured_inputs,
        'output': captured_outputs
    }



