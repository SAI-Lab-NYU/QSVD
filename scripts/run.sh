# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llava.sh 1.5 0
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llava.sh 1.5 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llava.sh 1.5 13
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llava.sh 1.5 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llava.sh 1.5 3
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llava.sh 1.5 1
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llava.sh 1.0
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llava.sh 1.0 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llava.sh 1.0 3
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llava.sh 0.9
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llava.sh 0.9 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llava.sh 0.9 3
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava.sh 0.8
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava.sh 0.7
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava.sh 0.6
# | ratio | seed | acc-sqa | W4A4 LLaVA v1.5 7B
# | 1.5   |   0  |  48.83  |
# | 1.5   |  42  |  56.92  |
# | 1.5   |  13  |  52.31  |
# | 1.5   |  5   |  45.96  |
# | 1.5   |  3   |  56.72  |
# | 1.5   |  1   |  47.55  |

# export TRANSFORMERS_VERBOSITY=error
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.9 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.9 3
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.9 1

export TRANSFORMERS_VERBOSITY=error
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 3
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 1

# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 0
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 2
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 4
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 6
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 7
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 8
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 9
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 10
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 11
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanextqa.sh 0.9 12
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext_cache.sh 0.9 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext_cache.sh 0.9 1
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext_cache.sh 0.9 0
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext_cache.sh 0.8 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext_cache.sh 0.7 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext_cache.sh 0.6 5

# # bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.9 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 1.5 5

# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext_cache.sh 1.5 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext_cache.sh 1.5 1
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext_cache.sh 1.5 3

# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 1.5 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 1.5 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 1.5 3

## a100
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.9 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.9 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.9 1
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.8 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.7 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext.sh 0.6 5


# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 1.5 0
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 1.5 5

# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 0.9 0
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 0.9 5

# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 0.8 0
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/run_llavanext.sh 0.6 0

# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext13b.sh 0.9 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext13b.sh 0.9 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext13b.sh 0.9 0
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext13b.sh 0.9 13


# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.9 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.9 13
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.9 42

# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext13b.sh 0.8 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext13b.sh 0.7 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext13b.sh 0.6 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.8 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.7 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.6 42
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.9 0
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.9 1
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.9 2
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.9 3
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.9 4

# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.8 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.7 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.6 5

# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.8 0
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.7 0
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava13b.sh 0.6 0

# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext13b.sh 0.8 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext13b.sh 0.7 5
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llavanext13b.sh 0.6 5

# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/fp16_llama3_1b.sh
# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_1b_qkv.sh 0 128
# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_1b_qkv.sh 0 64
# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_1b_qkv.sh 0 32

# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_3b_qkv.sh 0 256
# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_3b_qkv.sh 0 128
# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_3b_qkv.sh 0 64
# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_3b_qkv.sh 0 32
# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_1b_qkv.sh

# # bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_3b_qkv.sh 0 256 0.95 
# # bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_3b_qkv.sh 0 256 0.8 
# # bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_3b_qkv.sh 0 256 0.7 
# # bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_3b_qkv.sh 0 256 0.6

# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_1b_qkv.sh 0 256 0.95 
# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_1b_qkv.sh 0 256 0.8 
# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_1b_qkv.sh 0 256 0.7 
# bash /scratch/yw6594/cf/vlm/quant/WSVD/script/_ours_dev/qsvd-llm/llama3/wsvd/wsvd_llama3_1b_qkv.sh 0 256 0.6

bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava.sh 0.9 42
bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava.sh 0.9 5
bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava.sh 0.9 13
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava.sh 0.8 0
# bash /scratch/yw6594/cf/vlm/quant/QSVD/scripts/fp16_llava.sh 0.7 0