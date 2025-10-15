#!/bin/bash
nums=()
# nums=(4 10 20 30 40) # text vicuna_v1-8b
# nums=(3) # text vicuna_v1-8b y7
# nums=(13 14 15 16 17 18 19 20 21 22 23) # vision vit-l
# nums=(12) # vision vit-l y5
# v13 t4
for num in "${nums[@]}"; do
echo now have $num
python -m llava.eval.model_vqa_science_custom \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers_remove/wox1/top10/v$num/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --tnumbers "[]" \
    --vnumbers "[$num]" \
    --tsnumbers "[]" \
    --vsnumbers "[]" \
    --useflash

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers_remove/wox1/top10/v$num/llava-v1.5-13b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers_remove/wox1/top10/v$num/llava-v1.5-13b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers_remove/wox1/top10/v$num/llava-v1.5-13b_result.json


# python -m llava.eval.model_vqa_science_custom \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder ./playground/data/eval/scienceqa/images/test \
#     --answers-file ./playground/data/eval/scienceqa/answers_remove/woy5/top1/t$num/llava-v1.5-13b.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --max_token 1024 \
#     --topk 1 \
#     --conv-mode vicuna_v1 \
#     --imageonly \
#     --tnumbers "[$num]" \
#     --vnumbers "[]" \
#     --tsnumbers "[]" \
#     --vsnumbers "[]" \
#     --useflash

# python llava/eval/eval_science_qa.py \
#     --base-dir ./playground/data/eval/scienceqa \
#     --result-file ./playground/data/eval/scienceqa/answers_remove/woy5/top1/t$num/llava-v1.5-13b.jsonl \
#     --output-file ./playground/data/eval/scienceqa/answers_remove/woy5/top1/t$num/llava-v1.5-13b_output.jsonl \
#     --output-result ./playground/data/eval/scienceqa/answers_remove/woy5/top1/t$num/llava-v1.5-13b_result.json
done

python -m llava.eval.model_vqa_science_custom \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/ansers_WoutCO/L0_23cslmmean/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --max_token 1024 \
    --topk 1 \
    --conv-mode vicuna_v1 \
    --imageonly \
    --tnumbers "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]" \
    --vnumbers "[]" \
    --tsnumbers "[]" \
    --vsnumbers "[]" \
    --useflash
#     # --tnumbers "[4]" \ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
#     # --vnumbers "[]" \ [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
#     # --tsnumbers "[31]" \ 
#     # --vsnumbers "[]" \ 
python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/ansers_WoutCO/L0_23cslmmean/llava-v1.5-13b.jsonl \
    --output-file ./playground/data/eval/scienceqa/ansers_WoutCO/L0_23cslmmean/llava-v1.5-13b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/ansers_WoutCO/L0_23cslmmean/llava-v1.5-13b_result.json

# python -m llava.eval.model_vqa_science_custom \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder ./playground/data/eval/scienceqa/images/test \
#     --answers-file ./playground/data/eval/scienceqa/anwsers_RR/wox1/top10/v13/llava-v1.5-13b.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --max_token 1024 \
#     --topk 1 \
#     --conv-mode vicuna_v1 \
#     --imageonly \
#     --tnumbers "[]" \
#     --vnumbers "[13]" \
#     --tsnumbers "[]" \
#     --vsnumbers "[23]" \
#     --useflash
#     # --tnumbers "[4]" \
#     # --vnumbers "[]" \
#     # --tsnumbers "[31]" \
#     # --vsnumbers "[]" \
# python llava/eval/eval_science_qa.py \
#     --base-dir ./playground/data/eval/scienceqa \
#     --result-file ./playground/data/eval/scienceqa/anwsers_RR/wox1/top10/v13/llava-v1.5-13b.jsonl \
#     --output-file ./playground/data/eval/scienceqa/anwsers_RR/wox1/top10/v13/llava-v1.5-13b_output.jsonl \
#     --output-result ./playground/data/eval/scienceqa/anwsers_RR/wox1/top10/v13/llava-v1.5-13b_result.json

# python -m llava.eval.model_vqa_science_custom \
#     --model-path liuhaotian/llava-v1.5-13b \
#     --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder ./playground/data/eval/scienceqa/images/test \
#     --answers-file ./playground/data/eval/scienceqa/answers_remove/max_token=3/llava-v1.5-13b.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --conv-mode vicuna_v1 \
#     --max_token 3 \
#     --tnumbers "[]" \
#     --vnumbers "[]" \
#     --useflash \
#     # --imageonly \
    

# python llava/eval/eval_science_qa.py \
#     --base-dir ./playground/data/eval/scienceqa \
#     --result-file ./playground/data/eval/scienceqa/answers_remove/max_token=3/llava-v1.5-13b.jsonl \
#     --output-file ./playground/data/eval/scienceqa/answers_remove/max_token=3/llava-v1.5-13b_output.jsonl \
#     --output-result ./playground/data/eval/scienceqa/answers_remove/max_token=3/llava-v1.5-13b_result.json \