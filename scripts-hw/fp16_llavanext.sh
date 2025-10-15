export HF_HOME='/vast/hw3689/huggingface'
cd /scratch/hw3689/QSVD/fake_quant

source /vast/hw3689/miniforge3/bin/activate QSVD

# seed=(25)  #(20 21 22 23 24 25 26 27 28 29 30)
# wbits=16
# bits=16
# aclipratio=0.9
# bs=256
# svd_mode="UV"
# rank_ratio=${1:-0.9}

# for seed in "${seed[@]}"; do
# python mainllavanext.py \
#     --model llava-hf/llava-v1.6-vicuna-7b-hf  \
#     --a_bits "$bits" \
#     --w_bits "$wbits" \
#     --k_bits 16 \
#     --v_bits 16 \
#     --cal_dataset ScienceQA_Train \
#     --eval_dataset ScienceQA_TEST \
#     --tasks None \
#     --w_rtn \
#     --w_clip \
#     --a_clip_ratio "$aclipratio" \
#     --nsamples "$bs" \
#     --vitnsamples "$bs" \
#     --seed "$seed" \
#     --svd_mode "$svd_mode" \
#     --qkv_fuse \
#     --calib_method 'abs_mean' \
#     --rank_ratio "$rank_ratio" \
#     --act_aware \
#     --had_rank \
#     --svd_lm \
#     --act_alpha 0.5 \
#     --label_mode 'qa-qa' \
#     --basepath "/scratch/hw3689/QSVD" \
#     --setting "QSVD/sqa/qkvlm_svdgrad_h100/labelqaqa_fp32_0_-1/llavaaclip${aclipratio}_ratio${rank_ratio}${svd_mode}_mean${bs}_alpha=0.5_beta${beta_lr}_${beta_epochs}_bs${bs}/seed${seed}" \
#     --grad_info \
#     --beta_then_svd \
#     --cache_in_log 
# done

# v4 rtx haiyu gradinfo

seed=25 #(0 1 2 3 4 5 6 7 8 9)
wbits=16
bits=16
aclipratio=0.9
bs=256
svd_mode="UV"
rank_ratio=(0.6)

for rank_ratio in "${rank_ratio[@]}"; do
python mainllavanext.py \
    --model llava-hf/llava-v1.6-vicuna-7b-hf  \
    --a_bits "$bits" \
    --w_bits "$wbits" \
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
    --basepath "/scratch/hw3689/QSVD" \
    --setting "QSVD/sqa/qkvlm_svdgrad_h100/labelqaqa_fp32_0_-1/llavaaclip${aclipratio}_ratio${rank_ratio}${svd_mode}_mean${bs}_alpha=0.5_beta${beta_lr}_${beta_epochs}_bs${bs}/seed${seed}" \
    --grad_info \
    --beta_then_svd
    #--cache_in_log 
done
