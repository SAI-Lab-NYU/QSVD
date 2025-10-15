# QSVD: Efficient Low-Rank Approximation for Unified Query-Key-Value Weight Compression in Low-Precision Vision-Language Models

This repository provides the official implementation of **QSVD**, a method for efficient low-rank approximation that unifies Query-Key-Value (QKV) weight compression in low-precision Vision-Language Models (VLMs).

![QSVD Overview](figs/qsvd_overview.svg)

## 🌟 Highlights

- **🧩 Joint QKV Decomposition:**  
  QSVD performs a *unified singular value decomposition* on the concatenated query–key–value weight matrices  $[W_q, W_k, W_v]$, sharing a common down-projection $W_{qkv}^{d}$.  
  → Reduces parameters, low-rank KV-cache, and FLOPs compared to per-matrix SVD.

- **📊 Cross-Layer Rank Allocation:**  
  Introduces a *singular-value-wise importance analysis* to allocate ranks across layers based on each singular value’s contribution to model loss, enabling fine-grained, gradient-guided truncation.  
  → Preserves critical components while truncating redundant ones across all layers.

- **🎯 Post-Training Quantization for Low-Rank VLMs:**  
  Combines *dual orthogonal rotations* $(H_1, \ H_2)$ to smooth channel-wise outliers in both activations and latent buffers, together with an *adaptive exponent β* that rescales singular values to balance channel distributions and reduce quantization error.  
  → Jointly suppresses activation variance and outlier amplification, enabling stable low-precision inference with minimal degradation.


## 🔧 Requirements

This implementation utilize the myllava repository, adapted from the original [LLaVA repo](https://github.com/haotian-liu/LLaVA). Please follow the steps below to set up the environment:

```bash
git submodule update --init --recursive
conda create -n QSVD python=3.10 -y
conda activate QSVD
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install --no-build-isolation -r requirements.txt
```

<!-- > ⚠️ Note: Ensure the QSVD components and any relevant QuaRot setup are reinstalled correctly. -->

## 📊 Evaluation

To evaluate QSVD and reproduce our results, follow the steps below.

### 📁 Dataset Preparation

Follow the [LLaVA evaluation guide](https://github.com/haotian-liu/LLaVA/blob/main/docs/Evaluation.md) to prepare the following datasets:
- **ScienceQA** (Train) [LLaVA ScienceQA train](myllava/docs/QSVD_DATA.md)
- **VizWiz** (Test)

Update the paths in `eval_*.py` and `data_utils.py` accordingly.

### 🛠 Evaluation Toolkit

We use [third_party/VLMEvalKit](https://github.com/open-compass/VLMEvalKit/blob/main/docs/en/Quickstart.md) for evaluation. Set it up as per their quickstart guide.

### ▶️ Running Evaluations

To reproduce our main results:

For example usage and custom evaluations, explore the scripts in [fakequant](fake_quant/README.md):

or use scripts under

```bash
# download cache from huggingface
export HF_HOME='your_hf_home'
cd path_to_QSVD/fake_quant
conda activate QSVD
bash path_to_QSVD/scripts/fp16_cache_llavanext.sh 0.9 0 path_to_hf_cache path_to_QSVD
```

## 🤝 Contributing

This project builds upon the excellent work of:
- [QuaRot](https://github.com/spcl/QuaRot)
- [ASVD](https://github.com/hahnyuan/ASVD4LLM)
- [SVDLLM](https://github.com/AIoT-MLSys-Lab/SVD-LLM)

We thank these projects for their contributions to the community.
