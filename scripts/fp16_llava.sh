export HF_HOME='/vast/yw6594/log'
cd /scratch/yw6594/cf/vlm/quant/QSVD/fake_quant

source /vast/yw6594/miniforge3/bin/activate QSVD

wbits=16
bits=16
aclipratio=0.9
bs=256
svd_mode="U"
rank_ratio=${1:-0.9}
seed=${2:-0}
python mainllava.py \
    --model liuhaotian/llava-v1.5-7b  \
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
    --basepath "/scratch/yw6594/cf/vlm/quant/QSVD" \
    --setting "QSVD/sqa/qkvlm_svdgrad_ITv5_a100/labelqaqa/llavaaclip${aclipratio}_ratio${rank_ratio}${svd_mode}_mean${bs}_alpha=0.5_beta${beta_lr}_${beta_epochs}_bs${bs}/seed${seed}" \
    --grad_info \
    --beta_then_svd \
    --cache_in_log 


# v4 rtx haiyu gradinfo