import utils
import torch
import torch.nn as nn
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import hadamard_utils
import svd_utils
import profile_utils
import logging
import os
import torch.distributed as dist
import datetime

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size


@torch.no_grad()
def _profile(dataloader, args, model, image_processor, tokenizer, out=False):
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.vitnsamples, model.model.vision_tower.num_patches, model.model.vision_tower.config.hidden_size), dtype=dtype, device=utils.get_dev()
    )
    inps = []
    outs = torch.zeros(
        (args.vitnsamples, model.model.vision_tower.num_patches, model.model.vision_tower.config.hidden_size*5), dtype=dtype, device=utils.get_dev()
    ) if out else None
    cache = {'i': 0, 'attention_mask': [], 'position_embeddings':[]}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            # inps[cache['i']] = inp
            inps.append(inp)
            cache['i'] += 1
            raise ValueError
    class Catcherout(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, *args, **kwargs):
            inps[cache['i']] = inp
            inp = self.module(inp)
            outs[cache['i']] = inp
            cache['i'] += 1
            raise ValueError
    
    layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers
    model.model.vision_tower = model.model.vision_tower.to(utils.get_dev())
    layer = model.model.mm_projector
    layer = layer.to(utils.get_dev())

    
    if out:
        layer[0] = Catcherout(layer[0])
    else:
        layer[0] = Catcher(layer[0])

    for batch in dataloader:
        try:
            input_ids, images = gptq_utils.message_to_prompt(batch, image_processor, model, tokenizer)
            images, image_sizes = images
            from llava.constants import IMAGE_TOKEN_INDEX
            model.generate(input_ids, images=images,
                            image_sizes = image_sizes,
                            do_sample=False,
                            max_new_tokens=1,
                            use_cache=True,) # only infer till projector layer
        except ValueError:
            pass

    layer[0] = layer[0].module
    layer = layer.cpu()
    layers = layers.cpu()
    if out:
        return inps, outs
    return inps


def load_env():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, entity=args.wandb_id)
        wandb.config.update(args)
    utils.set_seed(args.seed)
    if args.vitnsamples == 0:
        args.vitnsamples = args.nsamples
        logging.info(f'set vitnsamples to nsamples{args.nsamples}')
    # transformers.set_seed(args.seed)
    return args


def main(args):
    
    model, tokenizer, image_processor = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    modeldtype = model.dtype
    model.model.vision_tower.to(modeldtype)
    print(model.model.vision_tower.dtype)
    if args.svd_lm and not args.beta_then_svd:
        # Start the Language Model part first, as svd depends on the concat input of vision and text
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor) 
        # utils.set_seed(args.seed)? 

    # utils.set_seed(args.seed)
    # dtype = next(iter(model.parameters())).dtype
    if args.vit_module:
        ### todo add vit learning whole process here
        if not args.vit_online:
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.vitnsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False, args=args
            )
            dataloader, _ = trainloader
            utils.set_seed(args.seed) # 
            del trainloader
            inpsfp16= _profile(dataloader, args, model, image_processor, tokenizer)
        config = model.model.vision_tower.config
        
        if args.rotate:
            # utils.set_seed(args.seed) # skip vit set ensure each component do not affect each other's seed
            if args.vit_online:
                rotation_utils.wrap_layer_normvit(model) # just ignore for now
                if args.mm_rh:
                    Q, Qmm = rotation_utils.rotate_modelvitmmonlineR_mmR(model, args)
                else:
                    Q = rotation_utils.rotate_modelvitmmonlineR(model, args)
                if args.vit_mmoff:
                    logging.info('mm proj rotation check again, skip them')
                utils.cleanup_memory(verbos=True)
            else:
                rotation_utils.fuse_layer_normsvit_returnskip(model)
                qlayers = model.model.vision_tower.vision_tower.vision_model # here add online hadamard for pre-ln
                qlayers.pre_layrnorm = model_utils.LNRotWrapper(qlayers.pre_layrnorm) # centering for vit
                # utils.set_seed(args.seed)
                if args.mm_rh:
                    Q, Qmm = rotation_utils.rotate_modelvitmmv2(model, args)
                else:
                    Q = rotation_utils.rotate_modelvitmmv2(model, args)
                utils.cleanup_memory(verbos=True)
            quant_utils.add_actquant(model.model.vision_tower.vision_tower.vision_model) #Add Activation Wrapper to the model
            
            quant_utils.add_actquant_mm(model.model)
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            
            for name in qlayers:# for clip, todo here
                if 'fc2' in name:
                    qlayers[name].online_random_had = True
                    qlayers[name].had_K = Q
                    qlayers[name].K = 1
                    qlayers[name].fp32_had = args.fp32_had
                if 'out_proj' in name: # why need this if already done ov_rotate in rotate_model
                    had_K, K = hadamard_utils.get_hadK(config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].had_dim = config.hidden_size//config.num_attention_heads
                    qlayers[name].fp32_had = args.fp32_had
                if 'mm_projector.0' in name and args.vit_online and not args.vit_mmoff:
                    logging.info('use mmproj input[0] online rotation')
                    # qlayers[name].online_random_had = True
                    # qlayers[name].had_K = Qmm
                    # qlayers[name].K = 1
                    # qlayers[name].fp32_had = args.fp32_had
                    had_K, K = hadamard_utils.get_hadK(model.config.mm_hidden_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                if 'mm_projector.2' in name and not args.vit_mmoff:
                    if args.mm_rh:# for now, we do not inplement fused yet
                        logging.info('use mmproj input[2] online RH rotation')
                        qlayers[name].online_random_had = True
                        qlayers[name].had_K = Qmm
                        qlayers[name].K = 1
                        qlayers[name].fp32_had = args.fp32_had
                    else:
                        logging.info('use mmproj input[2] online Had rotation')
                        had_K, K = hadamard_utils.get_hadK(model.config.hidden_size)
                        qlayers[name].online_full_had = True
                        qlayers[name].had_K = had_K
                        qlayers[name].K = K
                        qlayers[name].fp32_had = args.fp32_had
        else:
            quant_utils.add_actquant(model.model.vision_tower.vision_tower.vision_model) #Add Activation Wrapper to the model as the rest of the code assumes it is present
            quant_utils.add_actquant_mm(model.model)
        #### add vit weight quantization
        if args.w_bits < 16:
            save_dict = {}
            if args.load_qmodel_path: # Load Quantized Rotated Model
                assert args.rotate, "Model should be rotated to load a quantized model!"
                assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
                print("Load quantized model from ", args.load_qmodel_path)
                save_dict = torch.load(args.load_qmodel_path)
                model.load_state_dict(save_dict["model"])
                
            elif not args.w_rtn: # GPTQ Weight Quantization
                # assert ("llama" in args.model or 'llava' in args.model), "Only llama/llava is supported for GPTQ!"
                # utils.set_seed(args.seed)
                trainloader = data_utils.get_loaders(
                    args.cal_dataset, nsamples=args.nsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False, args=args
                )# use the same or not
                utils.set_seed(args.seed)
                quantizers = gptq_utils.gptq_fwrdvit(model, trainloader, utils.get_dev(), args, tokenizer, image_processor)
                save_dict["w_vitquantizers"] = quantizers
                if not args.vit_mmoff:
                    quantizers = gptq_utils.gptq_fwrdmm(model, trainloader, utils.get_dev(), args, tokenizer, image_processor)
                    save_dict["w_mmquantizers"] = quantizers
            else: # RTN Weight Quantization
                quantizers = gptq_utils.rtn_fwrdvit(model, utils.get_dev(), args)
                save_dict["w_vitquantizers"] = quantizers
                if not args.vit_mmoff:
                    quantizers = gptq_utils.rtn_fwrdmm(model, utils.get_dev(), args)
                    save_dict["w_mmquantizers"] = quantizers


        # Add vit Input Quantization
        if args.a_bits < 16 or args.v_bits < 16:
            logging.info(f'setting a clip ratio in vit {min(args.a_clip_ratio, args.vita_clip_ratio)}')
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            down_proj_groupsize = -1
            if args.a_groupsize > 0 and ("llama" in args.model or 'llava' in args.model): # here change as well  for all
                down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
            for name in qlayers:
                layer_input_bits = args.a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not(args.a_asym)
                layer_a_clip = min(args.a_clip_ratio, args.vita_clip_ratio)
                if 'lm_head' in name: #Skip lm_head quantization   
                    layer_input_bits = 16
                
                if 'fc2' in name: #Set the down_proj precision
                    if args.int8_down_proj:
                        layer_input_bits = 8
                    layer_groupsize = down_proj_groupsize

                if 'mm_projector' in name and args.vit_mmoff:
                    logging.info(f"mm projector act in {16} bit")
                    layer_input_bits = 16 # skip vit mm act quant
                qlayers[name].quantizer.configure(bits=layer_input_bits,
                                                groupsize=layer_groupsize,
                                                sym=layer_a_sym,
                                                clip_ratio=layer_a_clip)
        if not args.vit_online:
            inpsquant = _profile(dataloader, args, model, image_processor, tokenizer) 
        if world_size > 1:
            dist.barrier()
        if not args.vit_online and args.rotate:
            import beta_utils
            try:
                beta_path=None
                linearbias = torch.load(beta_path)
                logging.info("use beta from seed0 ep100 lowrank 71.83")
            except:
                weight = None
                bias_ = None
                Q = model.model.vision_tower.vision_tower.vision_model.pre_layrnorm.had_K
                linearbias = beta_utils.train_bias_linear(inpsfp16, inpsquant, args.beta_epochs, args.beta_lr, Q, args=args, weight=weight, bias=bias_)
            beta_utils.fuse_linearbias(linearbias, model, is_mm=True)
            del linearbias
            del inpsfp16
            del inpsquant
            utils.cleanup_memory(verbos=True)

    if args.svd_lm and args.beta_then_svd:
        # Start the Language Model part after vit setup, as svd depends on the concat input of vision and text
        # utils.set_seed(args.seed)
        svd_utils.svd_lm_setup(model, args, tokenizer, image_processor) 
        # torch.manual_seed(args.seed)
        # utils.set_seed(args.seed)
    if not args.lm_off:
        # Rotate the weights
        if args.rotate:
            model = model.cpu()
            # utils.set_seed(args.seed)
            if args.svd_lm:
                rotation_utils.fuse_layer_norms_noebsvdqkv(model)
                rotation_utils.rotate_modelllavasvdqkv(model, args)
            else:
                rotation_utils.fuse_layer_norms_noeb(model)
                rotation_utils.rotate_modelllava(model, args)
            utils.cleanup_memory(verbos=True)
            quant_utils.add_actquant(model.model.layers) #Add Activation Wrapper to only the LM model
            qlayers = quant_utils.find_qlayers(model)
            for name in qlayers:# for clip, todo here
                if 'down_proj' in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                    qlayers[name].online_full_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].fp32_had = args.fp32_had
                if 'o_proj' in name:
                    had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                    qlayers[name].online_partial_had = True
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
                    qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                    qlayers[name].fp32_had = args.fp32_had
        else:
            quant_utils.add_actquant(model.model.layers) #Add Activation Wrapper to the model as the rest of the code assumes it is present
        
        if args.profile_method:
            if args.vit_online:
                trainloader = data_utils.get_loaders(
                    args.cal_dataset, nsamples=args.vitnsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False, args=args
                )
                dataloader, _ = trainloader
                utils.set_seed(args.seed)
                del trainloader
            # layer_ids = [_ for _ in range(len(model.model.layers))]
            layer_ids = [0, 1, 2, 20, 21, 22, 37, 38, 39]
            profile_utils.save_layer(model, layer_ids, 'before_weight_quant', args)
            activations = {}
            # hooks = profile_utils.register_detailed_hooksv2(model, activations,) if not args.svd_lm else profile_utils.register_detailed_hooksv2svd(model, activations,)
            # torch.save(profile_utils._profile(dataloader, args, model, image_processor, tokenizer), f"{args.save_path}/before_quant_output.pth")
            hooks = profile_utils.register_detailed_hooksv2svdqkvout(model, activations)
            profile_utils._profile(dataloader, args, model, image_processor, tokenizer)
            torch.save(activations, f"{args.save_path}/before_quant_act.pth")
            del activations
            profile_utils.remove_hooks(hooks)
            # return
        
        if args.w_bits < 16:
            save_dict = {}
            if args.load_qmodel_path: # Load Quantized Rotated Model
                assert args.rotate, "Model should be rotated to load a quantized model!"
                assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
                print("Load quantized model from ", args.load_qmodel_path)
                save_dict = torch.load(args.load_qmodel_path)
                model.load_state_dict(save_dict["model"])
                
            elif not args.w_rtn: # GPTQ Weight Quantization
                assert ("llama" in args.model or 'llava' in args.model), "Only llama/llava is supported for GPTQ!"
                
                trainloader = data_utils.get_loaders(
                    args.cal_dataset, nsamples=args.nsamples,
                    seed=args.seed, model=args.model,
                    seqlen=model.seqlen, eval_mode=False, args=args
                )
                quantizers = gptq_utils.gptq_fwrdllava(model, trainloader, utils.get_dev(), args, tokenizer, image_processor)
                save_dict["w_quantizers"] = quantizers
            else: # RTN Weight Quantization
                quantizers = gptq_utils.rtn_fwrd(model, utils.get_dev(), args) #, start_id=start_id
                save_dict["w_quantizers"] = quantizers
                
            if args.save_qmodel_path:
                save_dict["model"] = model.state_dict()
                torch.save(save_dict, args.save_qmodel_path)

        if args.profile_method:
            pass
            # layer_ids = [_ for _ in range(len(model.model.layers))]
            # profile_utils.save_layer(model, layer_ids, 'after_weight_quant', args)
            # torch.save(profile_utils._profile(dataloader, args, model, image_processor, tokenizer), f"{args.save_path}/before_Aquant_output.pth")


        # Add Input Quantization
        if args.a_bits < 16 or args.v_bits < 16:
            logging.info(f'setting a clip ratio in lm {min(args.a_clip_ratio, args.lma_clip_ratio)}')
            qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
            down_proj_groupsize = -1
            if args.a_groupsize > 0 and "llama" in args.model:
                down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
            # ids  = ['.'+str(_)+'.' for _ in range(start_id)]
            for name in qlayers:
                # if any(id in name for id in ids):  # Skip names before start_id layer
                #     continue
                layer_input_bits = args.a_bits
                layer_groupsize = args.a_groupsize
                layer_a_sym = not(args.a_asym)
                layer_a_clip = min(args.a_clip_ratio, args.lma_clip_ratio)
                
                if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                    qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                                groupsize=args.v_groupsize,
                                                sym=not(args.v_asym),
                                                clip_ratio=args.v_clip_ratio)
                
                if 'lm_head' in name: #Skip lm_head quantization   
                    layer_input_bits = 16
                
                if 'down_proj' in name: #Set the down_proj precision
                    if args.int8_down_proj:
                        layer_input_bits = 8
                    layer_groupsize = down_proj_groupsize

                    
                qlayers[name].quantizer.configure(bits=layer_input_bits,
                                                groupsize=layer_groupsize,
                                                sym=layer_a_sym,
                                                clip_ratio=layer_a_clip)

        if args.k_bits < 16:
            if args.k_pre_rope:
                raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
            else:
                rope_function_name = model_utils.get_rope_function_name(model)
                layers = model_utils.get_layers(model)
                k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                            "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
                # for id, layer in enumerate(layers):
                #     if id < start_id:
                #         continue
                for layer in layers:
                    rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                                layer.self_attn, 
                                rope_function_name, 
                                config=model.config,
                                **k_quant_config)
                logging.info('finish qk rotation and k quant')
    
    if args.profile_method:
        activations = {}
        # hooks = profile_utils.register_detailed_hooksv2(model, activations,) if not args.svd_lm else profile_utils.register_detailed_hooksv2svd(model, activations,)
        # torch.save(profile_utils._profile(dataloader, args, model, image_processor, tokenizer), f"{args.save_path}/after_AWquant_output.pth")
        hooks = profile_utils.register_detailed_hooksv2svdqkvout(model, activations)
        profile_utils._profile(dataloader, args, model, image_processor, tokenizer)
        torch.save(activations, f"{args.save_path}/after_AWquant_act.pth")
        del activations
        profile_utils.remove_hooks(hooks)
        logging.info("Skip evaluation for profile calibrated results")
        return
    # Evaluating on dataset
    testloader = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True,
            args=args
        )

    if 'scienceqa' in args.eval_dataset.lower():
        import eval_utilsdistllava
        dataset_ppl = eval_utilsdistllava.evaluator(model, testloader, utils.get_dev(), args, tokenizer, image_processor)
    elif 'vizwiz' in args.eval_dataset.lower():
        import eval_utilsdistvizwiz
        dataset_ppl = eval_utilsdistvizwiz.evaluator(model, testloader, utils.get_dev(), args, tokenizer, image_processor)
    elif 'seedbench' in args.eval_dataset.lower():
        import eval_utilsdistseed
        dataset_ppl = eval_utilsdistseed.evaluator(model, testloader, utils.get_dev(), args, tokenizer, image_processor)
    else:
        dataset_ppl = eval_utilstransformersupgrade.evaluator(model, testloader, utils.get_dev(), args)
    if args.wandb:
            wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})


if __name__ == '__main__':
    args = load_env() # args/seeds setup before distributed setup?
    rank, world_size = get_rank_and_world_size()
    if world_size > 1:
        local_rank = os.environ.get('LOCAL_RANK', 0)
        torch.cuda.set_device(int(local_rank))
        dist.init_process_group(
            backend='nccl',
            timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
        )
    main(args)
    if world_size > 1:
        dist.destroy_process_group()