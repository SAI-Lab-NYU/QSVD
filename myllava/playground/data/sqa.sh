#!/bin/bash

# python -m llava.eval.model_vqa_science_llama \
#     --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \
#     --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
#     --image-folder ./playground/data/eval/scienceqa/images/test \
#     --answers-file ./playground/data/eval/scienceqa/answers_llamav/llava-v1.5-13b.jsonl \
#     --single-pred-prompt \
#     --temperature 0 \
#     --useflash \
    # --bfloat16 # comment to switch to float16 mode

# python llava/eval/eval_science_qa.py \
#     --base-dir ./playground/data/eval/scienceqa \
#     --result-file ./playground/data/eval/scienceqa/answers_llamav/llava-v1.5-13b.jsonl \
#     --output-file ./playground/data/eval/scienceqa/answers_llamav/llava-v1.5-13b_output.jsonl \
#     --output-result ./playground/data/eval/scienceqa/answers_llamav/llava-v1.5-13b_result.json

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-13b_result.json

# Total: 4241, Correct: 3176, Accuracy: 74.89%, IMG-Accuracy: 72.68% a100
# Total: 4241, Correct: 3176, Accuracy: 74.89%, IMG-Accuracy: 72.68% a100 [21:09<00:00,  3.34it/s]
# Total: 4241, Correct: 3178, Accuracy: 74.94%, IMG-Accuracy: 72.78% rtx8000 30min around
# Total: 4241, Correct: 3178, Accuracy: 74.94%, IMG-Accuracy: 72.78% rtx8000 30min around
# Total: 4241, Correct: 3176, Accuracy: 74.89%, IMG-Accuracy: 72.68% v100 30min around
# 72.78%
