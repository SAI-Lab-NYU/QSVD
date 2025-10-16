wbits=16
bits=16
aclipratio=0.9
bs=256
svd_mode="UV"
rank_ratio=${1:-1.5}
seed=${2:-0}
cache_file=${3:-"../cache_file/smolvlm"} # QSVD cache file path
database=${4:-"../"} # QSVD root path
python mainsmolvlm.py \
    --model HuggingFaceTB/SmolVLM-Instruct  \
    --a_bits "$bits" \
    --w_bits "$wbits" \
    --cal_dataset ScienceQA_Train \
    --eval_dataset ScienceQA_TEST \
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
    --cache_file "$cache_file" \
    --basepath "$database" \
    --setting "QSVD/sqa/cache/labelqaqa/smolvlmaclip${aclipratio}_ratio${rank_ratio}${svd_mode}_mean${bs}_alpha=0.5_beta${beta_lr}_${beta_epochs}_bs${bs}/seed${seed}" \
    --grad_info \
    --beta_then_svd \
    --cache_in_log 