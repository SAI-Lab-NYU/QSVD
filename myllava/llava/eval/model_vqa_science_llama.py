import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from accelerate import Accelerator
from PIL import Image as PIL_Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor

from peft import PeftModel
import gradio as gr
from huggingface_hub import HfFolder
import math
from typing import Callable, List, Optional, Sequence, Union, Tuple
# Initialize accelerator
accelerator = Accelerator()
device = accelerator.device

# Constants
DEFAULT_MODEL = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MAX_OUTPUT_TOKENS = 2048
MAX_IMAGE_SIZE = (1120, 1120)

def load_model_and_processor(model_name: str, finetuning_path: str = None, dtype=torch.bfloat16, useflash=False):
    """Load model and processor with optional LoRA adapter"""
    print(f"Loading model: {model_name}")
    hf_token = None
    if useflash:
        torch.backends.cuda.enable_flash_sdp(True)
    model = MllamaForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        use_safetensors=True,
        device_map=device,
        token=hf_token,
    )
    processor = MllamaProcessor.from_pretrained(model_name, token=hf_token, use_safetensors=True)

    if finetuning_path and os.path.exists(finetuning_path):
        print(f"Loading LoRA adapter from '{finetuning_path}'...")
        model = PeftModel.from_pretrained(
            model,
            finetuning_path,
            is_adapter=True,
            torch_dtype=dtype
        )
        print("LoRA adapter merged successfully")
    
    model, processor = accelerator.prepare(model, processor)
    return model, processor

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def process_image(image_path: str = None, image = None) -> PIL_Image.Image:
    """Process and validate image input"""
    if image is not None:
        return image.convert("RGB")
    if image_path and os.path.exists(image_path):
        return PIL_Image.open(image_path).convert("RGB")
    raise ValueError("No valid image provided")

def eval_model(args):
    # Model
    model_name = args.model_path
    dtype = torch.bfloat16 if args.bfloat16 else torch.float16
    print(f"Now using {dtype}")
    model, processor = load_model_and_processor(model_name,dtype=dtype,useflash=args.useflash)
    import ast
    tnumbers = ast.literal_eval(args.tnumbers)
    vnumbers = ast.literal_eval(args.vnumbers)
    vgnumbers = ast.literal_eval(args.vgnumbers)
    def register_custom_block_hooks(keys_to_mitigate=None):
        for idx, block in enumerate(model.vision_model.transformer.layers):
            if idx in vnumbers:
                enable_vision_custom_block(block, idx)
        for idx, block in enumerate(model.vision_model.global_transformer.layers):
            if idx in vgnumbers:
                enable_vision_custom_block(block, idx)
        for idx, block in enumerate(model.language_model.model.layers):
            if idx in tnumbers:
                enable_text_custom_block(block, idx)
    register_custom_block_hooks()
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0] # select the human
        qs = question['value'].replace('<image>', '').strip()
        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter directly."
        if 'image' in line:
            image_file = line["image"]
            image_path = os.path.join(args.image_folder, image_file)
            processed_image = process_image(image_path=image_path)
            conversation = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": qs}]}
            ]
        else:
            if args.imageonly:
                continue
            processed_image = None
            conversation = [
                {"role": "user", "content": [{"type": "text", "text": qs}]}
            ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(processed_image, prompt, return_tensors="pt").to(device)

        with torch.inference_mode():
            output = model.generate(
                **inputs, 
                # do_sample=True if args.temperature > 0 else False,
                # temperature=args.temperature,
                temperature=0.7, 
                top_p=0.9, 
                max_new_tokens=MAX_OUTPUT_TOKENS,
                use_cache=True,
            )
        if processed_image is not None:
            outputs = processor.decode(output[0], skip_special_tokens=True)[-2:-1].strip()
        else:
            outputs = processor.decode(output[0], skip_special_tokens=True)[-1:].strip()
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()


def replace_outlier_with_(act, key='median', threshold=200, topk = 3):
    act_ = act.abs()
    mean = torch.mean(act_, dim=(1, 2))
    median = torch.median(act_.flatten(1), dim=-1).values
    zero = torch.zeros_like(mean)
    topk_values, topk_indices = torch.topk(act_.flatten(1), topk)
    topk_indices = torch.unravel_index(topk_indices, act.shape[1:])
    for bs, (value, indice) in enumerate(zip(topk_values, zip(*topk_indices))):
        for v, idx in zip(value, zip(*indice)):
            if v > threshold * median[bs]:
                if key == 'median':
                    act[bs][idx] = median[bs]
                elif key == 'mean':
                    act[bs][idx] = mean[bs]
                else:
                    act[bs][idx] = zero[bs]


def visionblock_custom_forward(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = None,
    ):
        batchid = 0
        # print(f"norm1in  before change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
        # replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
        print(f"norm1in  after change max at batch{batchid}: {hidden_state[batchid].abs().max().detach()}")
        # replace_outlier_with_(hidden_states, 'mean') # here the input of norm1 x1
        residual = hidden_state
        hidden_state = self.input_layernorm(hidden_state)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm1 x2
        hidden_state, attn_weights = self.self_attn(hidden_state, attention_mask=attention_mask)
        if self.is_gated:
            hidden_state = self.gate_attn.tanh() * hidden_state
        hidden_state = residual + hidden_state

        # Feed forward
        # replace_outlier_with_(hidden_states, 'mean') # here the input of norm2 y1
        residual = hidden_state
        hidden_state = self.post_attention_layernorm(hidden_state)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm2 y2
        hidden_state = self.mlp(hidden_state)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of mlp y5
        if self.is_gated:
            hidden_state = self.gate_ffn.tanh() * hidden_state
        hidden_state = residual + hidden_state

        outputs = (hidden_state,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

def enable_vision_custom_block(block, block_id):
    block.block_id = block_id
    import types
    block.forward = types.MethodType(visionblock_custom_forward, block)

from transformers.cache_utils import Cache
def llamablock_custom_forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # replace_outlier_with_(hidden_states, 'mean') # here the input of norm1 x1
        batchid = 0
        # print(f"norm1in  before change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
        # replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
        print(f"norm1in  after change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm1 x2
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        # replace_outlier_with_(hidden_states, 'mean') # here the input of norm2 y1
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm2 y2
        hidden_states = self.mlp(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of mlp y7
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

def llamacrossblock_custom_forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # replace_outlier_with_(hidden_states, 'mean') # here the input of norm1 x1
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm1 x2
        # cross Attention
        hidden_states, attn_weights, past_key_value = self.cross_attn(
            hidden_states=hidden_states,
            attention_mask=cross_attention_mask,
            cross_attention_states=cross_attention_states,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

        # Fully Connected
        # replace_outlier_with_(hidden_states, 'mean') # here the input of norm2 y1
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm2 y2
        hidden_states = self.mlp(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of mlp y7
        if full_text_row_masked_out_mask is not None:
            hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states  # type: ignore
        hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states


        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        if use_cache:
            outputs += (past_key_value,)

        return outputs

def enable_text_custom_block(block, block_id):
    cross_idx = [3, 8, 13, 18, 23, 28, 33, 38]
    block.block_id = block_id
    import types
    if block_id in cross_idx:
        block.forward = types.MethodType(llamacrossblock_custom_forward, block)
    else:
        block.forward = types.MethodType(llamablock_custom_forward, block)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--bfloat16", action="store_true")
    parser.add_argument("--useflash", action="store_true")
    parser.add_argument("--imageonly", action="store_true")
    parser.add_argument('--tnumbers', type=str, default="[]", help="blocks to replace MA in text/fusion model")
    parser.add_argument('--vnumbers', type=str, default="[]", help="blocks to replace MA in vision model")
    parser.add_argument('--vgnumbers', type=str, default="[]", help="blocks to replace MA in global vision model")
    # details of model architecture refer to llama 3.2 vision -instruct model card.
    ### TODO: determine the threshold, and think about channel wise removal
    args = parser.parse_args()

    eval_model(args)
