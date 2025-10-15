import datasets
import random
import transformers

def get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
        
    if eval_mode:
        testdata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
        testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
        trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')    
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

def get_c4_new(nsamples, seed, seqlen, model, hf_token=None, eval_mode=False):

    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)

    if eval_mode:
        valdata = datasets.load_dataset(
        'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
        valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
        valenc = valenc.input_ids[:, :(256 * seqlen)]
        class TokenizerWrapper:
            def __init__(self, input_ids):
                self.input_ids = input_ids
        valenc = TokenizerWrapper(valenc)
        return valenc
    else:
        traindata = datasets.load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
        
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
                if trainenc.input_ids.shape[1] >= seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader

    


def get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode=False):
    
        
    if hf_token is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False, use_auth_token=hf_token)
    
    if eval_mode:
        testdata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='test')
        testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
        return testenc
    else:
        traindata = datasets.load_dataset('ptb_text_only', 'penn_treebank', split='train')
        trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader


def get_loaders(
    name, nsamples=128, seed=0, seqlen=2048, model='', hf_token=None, eval_mode=False, args=None
):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'ptb' in name:
        return get_ptb_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'c4' in name:
        return get_c4_new(nsamples, seed, seqlen, model, hf_token, eval_mode)
    if 'scienceqa' in name.lower():
        return get_vlmdata(name, nsamples, seed, seqlen, model, hf_token, args)
    if 'vizwiz' in name.lower():
        return None
    if 'seedbench' in name.lower():
        return get_vlmdata(name, nsamples, seed, seqlen, model)


def get_vlmdata(name, nsamples, seed, seqlen, model, hf_token=None, args=None):
    from vlmeval.dataset import build_dataset
    import pandas as pd
    if 'train' in name.lower():
        import json
        import os
        import math
        import logging
        logging.info('no anw with option in train prompt')
        def build_prompt_llava_train(line, pathbase):
            question = line['conversations'][0]
            question = question['value'].replace('<image>', '').strip()
            tgt_path = [pathbase + line["image"]]
            anwser = line['conversations'][1]['value'].strip()
            message = [dict(type="image", value=s) for s in tgt_path]
            message.append(dict(type="text", value=question))
            message.append(dict(type="textanw", value=anwser))
            return message
        def split_list(lst, n):
            """Split a list into n (roughly) equal-sized chunks"""
            chunk_size = math.ceil(len(lst) / n)  # integer division
            return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

        def get_chunk(lst, n, k):
            chunks = split_list(lst, n)
            return chunks[k]
        try:
            question_file = args.basepath + '/myllava/playground/data/eval/scienceqa/llava_train_QCM-LEA.json'
            image_base = args.basepath + '/myllava/playground/data/eval/scienceqa/images/train/' 
            dataset = json.load(open(os.path.expanduser(question_file), "r"))
        except:
            print('setup and change your datapath following llava evaluation.md')
        num_chunks = 1
        chunk_idx = 0
        dataset = get_chunk(dataset, num_chunks, chunk_idx)
        dataloader = []
        random.seed(seed)
        for _ in range(nsamples):
            i  =random.randint(0, len(dataset) - 1)
            line = dataset[i]
            dataloader.append(build_prompt_llava_train(line, image_base))
        return dataloader, None

    dataset = build_dataset(name)
    dataloader = []
    def build_prompt_llava(line, dataset):
        tgt_path = dataset.dump_image(line)
        question = line['question']
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question 
        question += "\nAnswer with the option's letter from the given choices directly."
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=question))
        return message
    for _ in range(0):
        i  =random.randint(0, len(dataset) - 1)
        data = dataset.data.iloc[i]
        dataloader.append(build_prompt_llava(data, dataset))
    return dataloader, dataset

def get_vizwiz_data(name, nsamples, seed, seqlen, model):
    from vlmeval.dataset import build_dataset
    import pandas as pd

    dataset = build_dataset(name)
    return dataset

def get_seedbench_data(name, nsamples, seed, seqlen, model):
    from vlmeval.dataset import build_dataset
    import pandas as pd
    dataset = build_dataset('SEEDBench_IMG')
    return dataset

def process_data(data, image_processor, model, tokenizer):
    from llava.mm_utils import (
        process_images,
        tokenizer_image_token,
        KeywordsStoppingCriteria,
    )
    from llava.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IMAGE_TOKEN_INDEX
    from PIL import Image
    import torch
    
    system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
    def concat_tilist(message):
        anw, text, images = "", "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
            elif item["type"] == "textanw":
                anw += item["value"]
        return text, images, anw
    
    # 处理批次数据
    batch_input_ids = []
    batch_image_tensors = []
    batch_answers = []
    
    for batch_item in data:
        content, images, anw = concat_tilist(batch_item)
        images = [Image.open(s).convert("RGB") for s in images]
        
        prompt = system_prompt + "USER: " + content + " ASSISTANT: "
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        answer_ids = tokenizer_image_token(anw, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        
        batch_input_ids.append(input_ids)
        
        if images:
            img_tensor = process_images(images, image_processor, model.config)
            batch_image_tensors.append(img_tensor)
        else:
            batch_image_tensors.append(None)
            
        batch_answers.append(answer_ids)
    
    return batch_input_ids, batch_image_tensors, batch_answers