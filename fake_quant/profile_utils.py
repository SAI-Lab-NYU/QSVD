import torch
import torch.nn as nn
import utils
import gptq_utils


def register_detailed_hooksv2svdqkvout(model, activations={}, keys_to_include=None):
    hooks = []

    def save_and_register(layer, name, *args, **kwargs):
        hook = layer.register_forward_hook(save_activationio(name, *args, activations=activations, **kwargs))
        hooks.append(hook)
    # for idx, block in enumerate(model.model.vision_tower.vision_tower.vision_model.encoder.layers):
    for idx in []:#[0, 1, 2, 3, 11, 12, 13, 14, 19, 20, 21, 23]:
        block = model.model.vision_tower.vision_tower.vision_model.encoder.layers[idx]

        ###### ln_1 in and out
        save_and_register(block.layer_norm1, f"llava.visual.block_{idx}.norm1", '_in', '_out')

        ###### attn module
        save_and_register(block.self_attn.q_proj.BLinear, f"llava.visual.block_{idx}.attn", out_key='q_proj')
        save_and_register(block.self_attn.k_proj.BLinear, f"llava.visual.block_{idx}.attn", out_key='k_proj')
        save_and_register(block.self_attn.v_proj.BLinear, f"llava.visual.block_{idx}.attn", out_key='v_proj')
        save_and_register(block.self_attn.out_proj.BLinear, f"llava.visual.block_{idx}.attn", out_key='out_proj_mid')
        save_and_register(block.self_attn.out_proj, f"llava.visual.block_{idx}.attn", in_key='_score', out_key='out_proj')

        ###### FFN ln_2 in and out
        save_and_register(block.layer_norm2, f"llava.visual.block_{idx}.norm2", '_in', '_out')

        # ###### FFN Gelu and mlp
        save_and_register(block.mlp.fc1.BLinear, f"llava.visual.block_{idx}.fc1")
        save_and_register(block.mlp.fc2.BLinear, f"llava.visual.block_{idx}.fc2")
        save_and_register(block.mlp.activation_fn, f"llava.visual.block_{idx}.gelu")

    print("Setting up hook for llava vision model")
    # for idx in [_ for _ in range(32)]:
    for idx in [0, 1, 2, 20, 21, 22, 37, 38, 39]:
        block = model.model.layers[idx]

        ###### rms_1 in and out
        save_and_register(block.input_layernorm, f"llava.text.block_{idx}.norm1", '_in', '_out')

        ###### attn module
        save_and_register(block.self_attn.q_proj.ALinear, f"llava.text.block_{idx}.attn", out_key='q_proj')
        save_and_register(block.self_attn.k_proj.ALinear, f"llava.text.block_{idx}.attn", in_key='kv_cache', out_key='k_proj')
        save_and_register(block.self_attn.v_proj.ALinear, f"llava.text.block_{idx}.attn", in_key='kv_cache', out_key='v_proj')
        # try:
        #     save_and_register(block.self_attn.o_proj.module, f"llava.text.block_{idx}.attn", in_key='_score', out_key='out_proj')
        # except:
        #     save_and_register(block.self_attn.o_proj, f"llava.text.block_{idx}.attn", in_key='_score', out_key='out_proj')

        # ###### FFN rms_2 in and out
        # save_and_register(block.post_attention_layernorm, f"llava.text.block_{idx}.norm2", '_in', '_out')

        # ###### FFN Gelu and mlp
        # save_and_register(block.mlp.gate_proj, f"llava.text.block_{idx}.fc11")
        # save_and_register(block.mlp.up_proj, f"llava.text.block_{idx}.fc1")
        # save_and_register(block.mlp.act_fn, f"llava.text.block_{idx}.silu")
        # save_and_register(block.mlp.down_proj, f"llava.text.block_{idx}.fc2", '_in', '_out')

    print("Setting up hook for llava text model")

    # mm_projector
    # block = model.model.mm_projector
    # save_and_register(block[0], f"llava.mmproj.fc1")
    # save_and_register(block[1], f"llava.mmproj.gelu")
    # save_and_register(block[2], f"llava.mmproj.fc2")

    # print("Setting up hook for llava proj model")

    # vision preln
    # block = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    # save_and_register(block, f"llava.vision.preln", '_in', '_out')

    # print("Setting up hook for llava vision model preln")

    return hooks  # Return hooks list to be used for removal


def register_detailed_hooksv2svd(model, activations={}, keys_to_include=None):
    hooks = []

    def save_and_register(layer, name, *args, **kwargs):
        hook = layer.register_forward_hook(save_activationio(name, *args, activations=activations, **kwargs))
        hooks.append(hook)
    # for idx, block in enumerate(model.model.vision_tower.vision_tower.vision_model.encoder.layers):
    for idx in []:#[0, 1, 2, 3, 11, 12, 13, 14, 19, 20, 21, 23]:
        block = model.model.vision_tower.vision_tower.vision_model.encoder.layers[idx]

        ###### ln_1 in and out
        save_and_register(block.layer_norm1, f"llava.visual.block_{idx}.norm1", '_in', '_out')

        ###### attn module
        save_and_register(block.self_attn.q_proj.BLinear, f"llava.visual.block_{idx}.attn", out_key='q_proj')
        save_and_register(block.self_attn.k_proj.BLinear, f"llava.visual.block_{idx}.attn", out_key='k_proj')
        save_and_register(block.self_attn.v_proj.BLinear, f"llava.visual.block_{idx}.attn", out_key='v_proj')
        save_and_register(block.self_attn.out_proj.BLinear, f"llava.visual.block_{idx}.attn", out_key='out_proj_mid')
        save_and_register(block.self_attn.out_proj, f"llava.visual.block_{idx}.attn", in_key='_score', out_key='out_proj')

        ###### FFN ln_2 in and out
        save_and_register(block.layer_norm2, f"llava.visual.block_{idx}.norm2", '_in', '_out')

        # ###### FFN Gelu and mlp
        save_and_register(block.mlp.fc1.BLinear, f"llava.visual.block_{idx}.fc1")
        save_and_register(block.mlp.fc2.BLinear, f"llava.visual.block_{idx}.fc2")
        save_and_register(block.mlp.activation_fn, f"llava.visual.block_{idx}.gelu")

    print("Setting up hook for llava vision model")
    # for idx in [_ for _ in range(32)]:
    for idx in [0, 1, 2, 20, 21, 22, 37, 38, 39]:
        block = model.model.layers[idx]

        ###### rms_1 in and out
        save_and_register(block.input_layernorm, f"llava.text.block_{idx}.norm1", '_in', '_out')

        ###### attn module
        save_and_register(block.self_attn.q_proj.BLinear, f"llava.text.block_{idx}.attn", out_key='q_proj')
        save_and_register(block.self_attn.k_proj.BLinear, f"llava.text.block_{idx}.attn", out_key='k_proj')
        save_and_register(block.self_attn.v_proj.BLinear, f"llava.text.block_{idx}.attn", out_key='v_proj')
        # try:
        #     save_and_register(block.self_attn.o_proj.module, f"llava.text.block_{idx}.attn", in_key='_score', out_key='out_proj')
        # except:
        #     save_and_register(block.self_attn.o_proj, f"llava.text.block_{idx}.attn", in_key='_score', out_key='out_proj')

        # ###### FFN rms_2 in and out
        # save_and_register(block.post_attention_layernorm, f"llava.text.block_{idx}.norm2", '_in', '_out')

        # ###### FFN Gelu and mlp
        # save_and_register(block.mlp.gate_proj, f"llava.text.block_{idx}.fc11")
        # save_and_register(block.mlp.up_proj, f"llava.text.block_{idx}.fc1")
        # save_and_register(block.mlp.act_fn, f"llava.text.block_{idx}.silu")
        # save_and_register(block.mlp.down_proj, f"llava.text.block_{idx}.fc2", '_in', '_out')

    print("Setting up hook for llava text model")

    # mm_projector
    # block = model.model.mm_projector
    # save_and_register(block[0], f"llava.mmproj.fc1")
    # save_and_register(block[1], f"llava.mmproj.gelu")
    # save_and_register(block[2], f"llava.mmproj.fc2")

    # print("Setting up hook for llava proj model")

    # vision preln
    # block = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    # save_and_register(block, f"llava.vision.preln", '_in', '_out')

    # print("Setting up hook for llava vision model preln")

    return hooks  # Return hooks list to be used for removal


def register_detailed_hooksv2(model, activations={}, keys_to_include=None):
    hooks = []

    def save_and_register(layer, name, *args, **kwargs):
        hook = layer.register_forward_hook(save_activationio(name, *args, activations=activations, **kwargs))
        hooks.append(hook)
    # for idx, block in enumerate(model.model.vision_tower.vision_tower.vision_model.encoder.layers):
    for idx in []:#[0, 1, 2, 3, 11, 12, 13, 14, 19, 20, 21, 23]:
        block = model.model.vision_tower.vision_tower.vision_model.encoder.layers[idx]

        ###### ln_1 in and out
        save_and_register(block.layer_norm1, f"llava.visual.block_{idx}.norm1", '_in', '_out')

        ###### attn module
        save_and_register(block.self_attn.q_proj, f"llava.visual.block_{idx}.attn", out_key='q_proj')
        save_and_register(block.self_attn.k_proj, f"llava.visual.block_{idx}.attn", out_key='k_proj')
        save_and_register(block.self_attn.v_proj, f"llava.visual.block_{idx}.attn", out_key='v_proj')
        save_and_register(block.self_attn.out_proj, f"llava.visual.block_{idx}.attn", in_key='_score', out_key='out_proj')

        ###### FFN ln_2 in and out
        save_and_register(block.layer_norm2, f"llava.visual.block_{idx}.norm2", '_in', '_out')

        # ###### FFN Gelu and mlp
        save_and_register(block.mlp.fc1, f"llava.visual.block_{idx}.fc1")
        save_and_register(block.mlp.fc2, f"llava.visual.block_{idx}.fc2")
        save_and_register(block.mlp.activation_fn, f"llava.visual.block_{idx}.gelu")

    print("Setting up hook for llava vision model")
    for idx in [_ for _ in range(32)]:
    # for idx in [0, 1, 2, 20, 21, 22, 37, 38, 39]:
        block = model.model.layers[idx]

        ###### rms_1 in and out
        save_and_register(block.input_layernorm, f"llava.text.block_{idx}.norm1", '_in', '_out')

        ###### attn module
        save_and_register(block.self_attn.q_proj, f"llava.text.block_{idx}.attn", out_key='q_proj')
        save_and_register(block.self_attn.k_proj, f"llava.text.block_{idx}.attn", out_key='k_proj')
        save_and_register(block.self_attn.v_proj, f"llava.text.block_{idx}.attn", out_key='v_proj')
        # try:
        #     save_and_register(block.self_attn.o_proj.module, f"llava.text.block_{idx}.attn", in_key='_score', out_key='out_proj')
        # except:
        #     save_and_register(block.self_attn.o_proj, f"llava.text.block_{idx}.attn", in_key='_score', out_key='out_proj')

        # ###### FFN rms_2 in and out
        # save_and_register(block.post_attention_layernorm, f"llava.text.block_{idx}.norm2", '_in', '_out')

        # ###### FFN Gelu and mlp
        # save_and_register(block.mlp.gate_proj, f"llava.text.block_{idx}.fc11")
        # save_and_register(block.mlp.up_proj, f"llava.text.block_{idx}.fc1")
        # save_and_register(block.mlp.act_fn, f"llava.text.block_{idx}.silu")
        # save_and_register(block.mlp.down_proj, f"llava.text.block_{idx}.fc2", '_in', '_out')

    print("Setting up hook for llava text model")

    # mm_projector
    # block = model.model.mm_projector
    # save_and_register(block[0], f"llava.mmproj.fc1")
    # save_and_register(block[1], f"llava.mmproj.gelu")
    # save_and_register(block[2], f"llava.mmproj.fc2")

    # print("Setting up hook for llava proj model")

    # vision preln
    # block = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm
    # save_and_register(block, f"llava.vision.preln", '_in', '_out')

    # print("Setting up hook for llava vision model preln")

    return hooks  # Return hooks list to be used for removal


def save_activationio(name, in_key=None, out_key='', activations={}):
    def hook(model, input, output):
        # Handle input (which is a tuple of tensors, we take first element)
        if in_key is not None:
            # For multi-batch input, we concatenate along batch dimension
            input_tensor = input[0].detach().clone().cpu()
            if name+in_key in activations:
                # If activation already exists, concatenate with new batch
                activations[name+in_key] = torch.cat([activations[name+in_key], input_tensor], dim=0)
            else:
                activations[name+in_key] = input_tensor
        
        # Handle output
        output_tensor = output.detach().clone().cpu()
        if name+out_key in activations:
            # If activation already exists, concatenate with new batch
            activations[name+out_key] = torch.cat([activations[name+out_key], output_tensor], dim=0)
        else:
            activations[name+out_key] = output_tensor
    return hook

def remove_hooks(hooks):
    """Removes all registered hooks"""
    for hook in hooks:
        hook.remove()
    print("All hooks removed.")


def save_layer(model, layer_ids, keys, args):    
    save_layers = dict()
    for idx in layer_ids:
        save_layers[f'{idx}'] = model.model.layers[idx].self_attn.k_proj
    torch.save(save_layers, f"{args.save_path}/{keys}.pth")



def _profile(dataloader, args, model, image_processor=None, tokenizer=None):
    
    inps = []
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings':[]}

    class Catcherout(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inp = self.module(inp, *args, **kwargs)
            # inps[cache['i']] = inp
            inps.append(inp[0].cpu())
            cache['i'] += 1
            # cache['attention_mask'].append(kwargs['attention_mask'].cpu())
            # cache['position_embeddings'].append(kwargs['position_embeddings'])
            raise ValueError
    model.model = model.model.to(utils.get_dev())
    try:
        model.model.layers[39] = Catcherout(model.model.layers[39])
    except:
        model.model.layers[31] = Catcherout(model.model.layers[31])
    for batch in dataloader:
        try:
            # model(batch[0].to(dev))
            input_ids, images = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
            images, image_size = images
            from llava.constants import IMAGE_TOKEN_INDEX
            # num_images = (input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images)
            model.generate(input_ids, images=images,
                            do_sample=False,
                            temperature=0,
                            max_new_tokens=512,
                            use_cache=True,)
        except ValueError:
            pass
        break # only forward once for hook?
    try:
        model.model.layers[39] = model.model.layers[39].module
    except:
        model.model.layers[31] = model.model.layers[31].module
    # self_attn.q_proj
    model.model = model.model.cpu()
    return inps