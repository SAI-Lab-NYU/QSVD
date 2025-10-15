# export HF_HOME='your_hf_home'
cd QSVD/fake_quant
seed=0
wbits=4
bits=4
aclipratio=0.9
bs=256
svd_mode="U"
rank_ratio=1.5
beta_lr=1.0
beta_epochs=100
torchrun --nproc_per_node 2 --master_port=$((seed + 2333)) mainsmolvlm.py \
    --model HuggingFaceTB/SmolVLM-Instruct  \
    --a_bits "$bits" \
    --w_bits "$wbits" \
    --k_bits 16 \
    --v_bits 16 \
    --cal_dataset ScienceQA_Train \
    --eval_dataset ScienceQA_TEST \
    --tasks None \
    --w_rtn \
    --rot_epochs 5 \
    --rot_lr 1e-3 \
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
    --setting "QSVD/sqa/online_then_qkvlm_svdgrad/labelqaqa/llavaaclip${aclipratio}_ratio${rank_ratio}${svd_mode}_mean${bs}_alpha=0.5_beta${beta_lr}_${beta_epochs}_bs${bs}/seed${seed}" \
    --beta_lr "$beta_lr" \
    --beta_epochs "$beta_epochs" \
    --rotate \
    --vit_module \
    --vit_online \
    --grad_info \
    --beta_then_svd \
    --cache_in_log 