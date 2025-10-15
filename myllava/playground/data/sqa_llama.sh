#!/bin/bash
# 2 13 14 vision
# 2_40 text

cd xxx/LLaVA
python -m llava.eval.model_vqa_science_llama \
    --model-path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers_llamav/print/llava-v1.5-13b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --tnumbers "[2]" \
    --vnumbers "[]" \
    --vgnumbers "[]" \
    --bfloat16 \
    --imageonly


python llava/eval/eval_science_qa_llama.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers_llamav/print/llava-v1.5-13b.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers_llamav/print/llava-v1.5-13b_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers_llamav/print/llava-v1.5-13b_result.json

##### 
# 1. much longer generation process
# 2. later generation forward do not have MA [300 -> 0.]
# 