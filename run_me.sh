source /data1/vikram/miniconda3/etc/profile.d/conda.sh
conda activate QSVD
cd /data1/vikram/QSVD/QSVD
python -u /data1/vikram/QSVD/QSVD/fake_quant/mainllavanext.py \
    --model llava-hf/llava-v1.6-vicuna-7b-hf  \
    --a_bits 4 \
    --w_bits 4 \
    --k_bits 16 \
    --v_bits 16 \
    --cal_dataset ScienceQA_Train \
    --eval_dataset ScienceQA_TEST \
    --tasks None \
    --w_rtn \
    --w_clip \
    --a_clip_ratio 0.9 \
    --nsamples 256 \
    --vitnsamples 256 \
    --seed 0 \
    --svd_mode 0.2 \
    --qkv_fuse \
    --calib_method 'abs_mean' \
    --rank_ratio 1.5 \
    --act_aware \
    --had_rank \
    --svd_lm \
    --act_alpha 0.5 \
    --label_mode 'qa-qa' \
    --basepath "/data1/vikram/QVLM/" \
    --setting "QSVD/sqa/llavanext_aclip0.9_ratio1.5_mean4" \
    --beta_lr 1.0 \
    --beta_epochs 100 \
    --rotate \
    --vit_module \
    --grad_info \
    --beta_then_svd
echo "PYTHON_EXIT_CODE=$?"
