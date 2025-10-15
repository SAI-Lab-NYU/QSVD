python scripts/convert_sqa_to_llava.py \
    convert_to_llava \
    --base-dir xxx/LLaVA/playground/data/eval/scienceqa \
    --prompt-format "QCM-LEA" \
    --split train
    # --split {train,val,minival,test,minitest}