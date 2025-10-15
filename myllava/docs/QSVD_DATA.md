### ScienceQA

#### Prepare Data
1. Please see ScienceQA [repo](https://github.com/lupantech/ScienceQA?tab=readme-ov-file#ghost-download-the-dataset) for setting up the dataset.
2. Generate ScienceQA dataset for LLaVA conversation-style format.

```Shell
cd QSVD
python myllava/scripts/convert_sqa_to_llava.py \
    convert_to_llava \
    --base-dir /path/to/ScienceQA/data/scienceqa \
    --prompt-format "QCM-LEA" \
    --split {train,val,minival,test,minitest}

```
