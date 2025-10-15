export HF_HOME='/vast/hw3689/huggingface'
cd /scratch/hw3689/QSVD/fake_quant

source /vast/hw3689/miniforge3/bin/activate QSVD

seed=(1 3 4)
wbits=16
bits=16
aclipratio=0.9
bs=256
svd_mode="UV"
rank_ratio=${1:-1.5}

for seed in "${seed[@]}"; do
python mainsmolvlm.py \
    --model HuggingFaceTB/SmolVLM-Instruct  \
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
    --setting "QSVD/sqa/qkvlm_svdgrad_h100/labelqaqa_fp32_0_-1_QSVDenv/smolvlmaclip${aclipratio}_ratio${rank_ratio}${svd_mode}_mean${bs}_alpha=0.5_beta${beta_lr}_${beta_epochs}_bs${bs}/seed${seed}" \
    --grad_info \
    --beta_then_svd
    #--cache_in_log 
done
