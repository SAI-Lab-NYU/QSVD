# Fake Quantization in QSVD


In this directory, we provide the torch scripts for the experiments in QSVD. 


## VQA Evaluations

Currently, we only support **llava-v1.5** models. You can simply run the `main.py` to reproduce the results in the paper. The most important arguments are:

- `--model`: the model name (or path to the weights)
- `--seed`: control the random seed
- `--nsamples`: the number of samples for SVD calibration 
- `--rotate`: whether we want to rotate the model (apply quarot)
- `--tasks`: the tasks for LM-Eval
- `--cal_dataset`: the calibration dataset for GPTQ quantization/SVD calibration
- `--eval_dataset`: Evaluation dataset
- `--a_bits`: the number of bits for activation quantization
- `--w_bits`: the number of bits for weight quantization
- `--v_bits`: the number of bits for value quantization (depracated if using SVD)
- `--k_bits`: the number of bits for key quantization (depracated if using SVD)
- `--w_clip`: Whether we want to clip the weights
- `--a_clip_ratio`: The ratio of clipping for activation
- `--vita_clip_ratio`: override The ratio of clipping for vit activation
- `--lma_clip_ratio`: override The ratio of clipping for language model activation
- `--k_clip_ratio`: The ratio of clipping for key (depracated if using SVD)
- `--v_clip_ratio`: The ratio of clipping for value  (depracated if using SVD)
- `--w_asym`: Whether we want to use asymmetric quantization for weights
- `--a_asym`: Whether we want to use asymmetric quantization for activation
- `--v_asym`: Whether we want to use asymmetric quantization for value
- `--k_asym`: Whether we want to use asymmetric quantization for key
- `--a_groupsize`: The group size for activation quantization
- `--w_groupsize`: The group size for weight quantization
- `--v_groupsize`: The group size for value quantization
- `--k_groupsize`: The group size for key quantization
- `--svd_mode`: Choose how sigma is fused in SVD weights
- `--qkv_fuse`: Whether we concact QKV proj for SVD
- `--calib_method`: Choose SVD method
- `--rank_ratio`: SVD rank ratio (rank = C_in * C_out / (C_in + C_out) * rank_ratio)
- `--act_aware`: Whether use activation aware SVD
- `--had_rank`: Whether add rotation in SVD latent activation
- `--svd_lm`: Whether we apply SVD
- `--act_alpha`: activation aware SVD related hyperparamter
- `--vit_module`: Whether we apply quantization in ViT
- `--grad_info`: Whether we use cross-layer rank allocation
- `--beta_then_svd`: Whether we apply SVD after ViT quantization

  
For example, to run the ScienceQA evaluation of `llava-v1.5-7b` model with quantizing all weights and activations, you can run the following command:

```bash
cd QSVD/fake_quant
python main.py --model liuhaotian/llava-v1.5-7b  \
                --a_bits 4 \
                --w_bits 4 \
                --k_bits 16 \
                --v_bits 16 \
                --cal_dataset ScienceQA_Train \
                --eval_dataset ScienceQA_TEST \
                --w_rtn \
                --w_clip \
                --lma_clip_ratio 0.9 \
                --nsamples 256 \
                --seed 0 \
                --svd_mode "U" \
                --qkv_fuse \
                --calib_method 'abs_mean' \
                --rank_ratio 1.5 \
                --act_aware \
                --had_rank \
                --svd_lm \
                --act_alpha 0.5 \
                --setting "/sqa/online_then_qkvlm_svdgrad/seed0" \
                --rotate \
                --vit_module \
                --vit_online \
                --grad_info \
                --beta_then_svd
```
