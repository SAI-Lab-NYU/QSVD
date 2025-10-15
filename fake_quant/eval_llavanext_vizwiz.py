import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import logging
import utils

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from llava.eval.m4c_evaluator import EvalAIAnswerProcessor

from PIL import Image
import math


import os
import torch.distributed as dist
import datetime

def get_rank_and_world_size():
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return rank, world_size

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def output_process(answer):
    if "<s>" in answer:
        answer = answer.replace("<s>", "").strip()
    if "[/INST]" in answer:
        answer = answer.split("[/INST]")[1].strip()
    elif "ASSISTANT:" in answer:
        answer = answer.split("ASSISTANT:")[1].strip()
    elif "assistant\n" in answer:
        answer = answer.split("assistant\n")[1].strip()
    elif "<|end_header_id|>\n\n" in answer:
        answer = answer.split("<|end_header_id|>\n\n")[2].strip()

    if "</s>" in answer:
        answer = answer.split("</s>")[0].strip()
    elif "<|im_end|>" in answer:
        answer = answer.split("<|im_end|>")[0].strip()
    elif "<|eot_id|>" in answer:
        answer = answer.split("<|eot_id|>")[0].strip()
    return answer

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __getitem__(self, index):
        line = self.questions[index]
        
        question_id = line["question_id"]
        image_file = line["image"]
        # Assuming 'text' is the question field in your JSONL
        question_text = line["text"]

        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"}, # Placeholder for the image
                    {"type": "text", "text": question_text}
                ]
            },
        ]
        prompt = self.image_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.image_processor(text=prompt, images=image, return_tensors="pt")
        
        return inputs

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    # 由于每个批次只有一个样本，直接返回第一个元素
    return batch[0]


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    return data_loader


def evaluator(model, testloader, device, args, tokenizer, image_processor):
    rank, world_size = get_rank_and_world_size()
    model_name = args.model
    num_chunks = world_size
    chunk_idx = rank
    # Data
    try:
        base_path = 'xxxxx'
        question_file = f'{base_path}/playground/data/eval/vizwiz/llava_test.jsonl'
        if not os.path.exists(question_file):
            raise FileNotFoundError(f"Question file not found: {question_file}")
    except FileNotFoundError as e:
        print('setup and change your datapath following llava evaluation.md')
    # base_path = '/home/why/LLaVA'
    # question_file = f'{base_path}/playground/data/eval/vizwiz/llava_test.jsonl'
    image_folder = f'{base_path}/playground/data/eval/vizwiz/test'
    answers_file = args.save_path + f'/vizwiz_answer-{rank}.jsonl'
    temperature = 0
    conv_mode = 'vicuna_v1'
    
    # convert_vizwiz_for_submission
    annotation_file = question_file
    
    result_upload_file = answers_file.replace('.jsonl', '_upload.json')
    #result_upload_file = './playground/data/eval/vizwiz/answers_upload/llava-v1.5-13b.json'

    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    # 尝试读取整个JSON文件
    # with open(os.path.expanduser(question_file), "r") as f:
    #     questions = json.load(f)
    #     # 如果questions不是列表，将其转换为列表
    #     if not isinstance(questions, list):
    #         questions = [questions]
    questions = get_chunk(questions, num_chunks, chunk_idx)
    existing_ids = set()
    if os.path.exists(answers_file):
        with open(answers_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    existing_ids.add(data['question_id'])
                except json.JSONDecodeError:
                    continue

    answers_file = os.path.expanduser(answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "a") # Append mode for resuming, or create new if not exists


    data_loader = create_data_loader(questions, image_folder, tokenizer, image_processor, model.config)

    # 确保模型在正确的设备上
    model = model.to(rank)
    logging.info(f"Model moved to device: {rank}")
    device = utils.get_dev()
    i = 0
    for inputs, line in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = line["question_id"]
        if idx in existing_ids:
            continue
        cur_prompt = line["text"]
        
        # 确保输入张量在正确的设备上
        inputs = {k: v.to(device=device, non_blocking=True) for k, v in inputs.items()}


        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                do_sample=True if temperature > 0 else False,
                temperature=temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=16,
                use_cache=True)

        outputs = image_processor.decode(outputs[0], skip_special_tokens=True)
        outputs = output_process(outputs)
        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        if i %10 == 0:
            ans_file.flush()
        i += 1
    ans_file.close()
    
    # merge the results
    if world_size > 1:
        dist.barrier()
    if rank == 0:
        import glob
        result_file = args.save_path + f'/vizwiz_answer.jsonl'
        # Clear or create the final merged file
        with open(result_file, 'w') as outfile:
            # Find all partial answer files
            answer_files = sorted(glob.glob(os.path.join(args.save_path, 'vizwiz_answer-*.jsonl')))
            
            for file in answer_files:
                with open(file, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
        result_upload_file = result_file.replace('.jsonl', '_upload.json')
        os.makedirs(os.path.dirname(result_upload_file), exist_ok=True)

        results = []
        error_line = 0
        for line_idx, line in enumerate(open(result_file)):
            try:
                results.append(json.loads(line))
            except:
                error_line += 1
        results = {x['question_id']: x['text'] for x in results}
        test_split = [json.loads(line) for line in open(annotation_file)]
        split_ids = set([x['question_id'] for x in test_split])

        print(f'total results: {len(results)}, total split: {len(test_split)}, error_line: {error_line}')

        all_answers = []

        answer_processor = EvalAIAnswerProcessor()

        for x in test_split:
            assert x['question_id'] in results
            all_answers.append({
                'image': x['image'],
                'answer': answer_processor(results[x['question_id']])
            })

        with open(result_upload_file, 'w') as f:
            json.dump(all_answers, f)
