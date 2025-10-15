import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import hadamard_utils
import os
import time
import act_aware_utils
from tqdm import tqdm

def main():
    args = utils.parser_gen()
    
    # Create directory to save model
    save_dir = os.path.join('../calibrated_models', args.setting)
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Loading model: {args.model}")
    transformers.set_seed(args.seed)
    model, tokenizer, image_processor = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    modeldtype = model.dtype
    model.model.vision_tower.to(modeldtype)
    
    # Rotate weights
    if args.rotate:
        print("Rotating model weights...")
        rotation_utils.fuse_layer_norms_noeb(model)
        rotation_utils.rotate_modelllava(model, args)
        utils.cleanup_memory(verbos=True)
            
        quant_utils.add_actquant(model.model.layers) # Only add activation quantization wrapper for LM model
        qlayers = quant_utils.find_qlayers(model)
        # for name in qlayers:
        #     if 'down_proj' in name:
        #         had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
        #         qlayers[name].online_full_had = True
        #         qlayers[name].had_K = had_K
        #         qlayers[name].K = K
        #         qlayers[name].fp32_had = args.fp32_had
        #     if 'o_proj' in name:
        #         had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
        #         qlayers[name].online_partial_had = True
        #         qlayers[name].had_K = had_K
        #         qlayers[name].K = K
        #         qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
        #         qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model) # Add activation quantization wrapper for model
    
    # Removed SVD compression part
    
    # Load calibration dataset
    print(f"Loading calibration dataset: {args.cal_dataset}")
    start_time = time.time()
    calib_loader = data_utils.get_loaders(
        args.cal_dataset, 
        nsamples=args.nsamples,
        seed=args.seed, 
        model=args.model,
        seqlen=model.seqlen, 
        eval_mode=False
    )
    print(f"Dataset loading completed, time taken: {time.time() - start_time:.2f} seconds")
    
    # Create cache directory
    os.makedirs("cache", exist_ok=True)
    
    # Calibrate using act_aware_utils
    print("Starting input distribution calibration...")
    for method in ["abs_mean", "abs_max"]:
        print(f"Calibrating with {method} method...")
        act_aware_utils.calib_input_distribution(model, tokenizer, image_processor, calib_loader, args, method=method, use_cache=False)
        
        # Save calibrated model separately for each method
        method_save_dir = os.path.join(save_dir, method)
        os.makedirs(method_save_dir, exist_ok=True)
        
        # Modify model save name, add rotate information
        rotate_info = "rotated" if args.rotate else "norotate"
        model_save_path = os.path.join(method_save_dir, f"{args.model.replace('/', '_')}_{rotate_info}_calibrated.pt")
        print(f"Saving model calibrated with {method} method to: {model_save_path}")
        
        save_dict = {
            "model": model.state_dict(),
            "config": model.config,
            "calibration_info": {
                "dataset": args.cal_dataset,
                "nsamples": args.nsamples,
                "rotate": args.rotate,
                "method": method,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        torch.save(save_dict, model_save_path)
        print(f"Calibration with {method} method completed! Model saved to: {model_save_path}")
        
        # Save calibration information text file for easy viewing
        info_path = os.path.join(method_save_dir, f"{args.model.replace('/', '_')}_calibration_info.txt")
        with open(info_path, 'w') as f:
            f.write(f"Model: {args.model}\n")
            f.write(f"Calibration dataset: {args.cal_dataset}\n")
            f.write(f"Number of samples: {args.nsamples}\n")
            f.write(f"Rotation: {args.rotate}\n")
            f.write(f"Calibration method: {method}\n")
            f.write(f"Calibration time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Calibration information for {method} method saved to: {info_path}")
    
    # Removed original save code, as each method is now saved separately
    print(f"All calibration methods completed!")

if __name__ == '__main__':
    main()