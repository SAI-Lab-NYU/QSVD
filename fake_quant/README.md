# Fake Quantization in QSVD


In this directory, we provide the torch scripts for the experiments in QSVD. 


## VQA Evaluations

Currently, we only support **SmolVLM, LLaVA-v1.5, LLaVA-Next** models. You can simply run the `mainsmolvlm.py`, `mainllava.py`, or `mainllavanext.py` accordingly to reproduce the results in the paper. The most important arguments are:

- `--model`: Model name (or path to the weights)
- `--seed`: Control the random seed
- `--nsamples`: Number of samples for SVD calibration 
- `--rotate`: Whether we want to rotate the model (apply quarot)
- `--tasks`: Tasks for LM-Eval
- `--cal_dataset`: Calibration dataset for GPTQ quantization/SVD calibration (currently support `ScienceQA_Train`)
- `--eval_dataset`: Evaluation dataset (currently support `ScienceQA_TEST` and `VizWiz`)
- `--a_bits`: Number of bits for activation quantization
- `--w_bits`: Number of bits for weight quantization
- `--v_bits`: Number of bits for value quantization (depracated if using SVD)
- `--k_bits`: Number of bits for key quantization (depracated if using SVD)
- `--w_clip`: Whether we want to clip the weights
- `--a_clip_ratio`: The ratio of clipping for activation
- `--vita_clip_ratio`: Override the ratio of clipping for vit activation
- `--lma_clip_ratio`: Override the ratio of clipping for language model activation
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
- `--qkv_fuse`: Whether we concact QKV for joint SVD proposed in our paper
- `--calib_method`: Choose SVD whitening method (`abs_max` and `abs_mean` for ASVD-style)
- `--rank_ratio`: 2 * SVD rank ratio (the factor of 2 is a legacy setting)
- `--act_aware`: Whether use activation aware SVD
- `--had_rank`: Whether add rotation (H₂ in our paper) in SVD latent activation 
- `--svd_lm`: Whether we apply SVD
- `--act_alpha`: Activation-aware SVD related hyperparamter of ASVD
- `--vit_module`: Whether we apply quantization in ViT
- `--grad_info`: Whether we use cross-layer rank allocation proposed in our paper
- `--beta_then_svd`: Whether we apply SVD after ViT quantization
- `--cache_file`: Path to pre-computed calibration cache file folder
- `--basepath`: Path to the parent folder of myllava (where we store ScienceQA and VizWiz dataset)

  
For example, to run the ScienceQA evaluation of `llava-v1.5-7b` model with quantizing all weights and activations, you can run the following command:

```bash
export HF_HOME='your_hf_home'
cd path_to_QSVD/fake_quant
conda activate QSVD
python mainllava.py --model liuhaotian/llava-v1.5-7b  \
                --a_bits 4 \
                --w_bits 4 \
                --k_bits 16 \
                --v_bits 16 \
                --cal_dataset ScienceQA_Train \
                --eval_dataset ScienceQA_TEST \
                --tasks None \
                --w_rtn \
                --w_clip \
                --a_clip_ratio "$aclipratio" \
                --nsamples "$bs" \
                --vitnsamples "$bs" \
                --seed "$seed" \
                --svd_mode "$svd_mode" \
                --qkv_fuse \
                --calib_method 'abs_mean' \
                --rank_ratio "$rank_ratio" \
                --act_aware \
                --had_rank \
                --svd_lm \
                --act_alpha 0.5 \
                --label_mode 'qa-qa' \
                --basepath "../" \
                --setting "/sqa/seed0" \
                --rotate \
                --cache_in_log \
                --grad_info \
                --beta_then_svd
```
or using existing scripts under path_to_QSVD/scripts, some example usage like
```
export HF_HOME='your_hf_home'
cd path_to_QSVD/fake_quant
conda activate QSVD
########## llava
# fp16
bash path_to_QSVD/scripts/fp16_llava.sh 0.9 
# w4a4
bash path_to_QSVD/scripts/run_llava.sh 1.5 
########## llava-next
# fp16
bash path_to_QSVD/scripts/fp16_llavanext.sh 0.9 
# w4a4
bash path_to_QSVD/scripts/run_llavanext.sh 1.5 
```
