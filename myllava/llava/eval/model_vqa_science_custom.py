import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from PIL import Image
import math
from typing import Optional,Tuple


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    if args.useflash:
        torch.backends.cuda.enable_flash_sdp(True)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name, attn_implementation='eager',)
    # print(str(model))
    # start logging block

    import ast
    tnumbers = ast.literal_eval(args.tnumbers)
    vnumbers = ast.literal_eval(args.vnumbers)
    vsnumbers = ast.literal_eval(args.vsnumbers)
    tsnumbers = ast.literal_eval(args.tsnumbers)
    attn_layers = tnumbers
    mlp_layers = tnumbers
    channels = [3050 ,4727 ,4743 ,2772 ,1843]
    def replace_parameter_CO_Woutlm(model, alayeridx, mlayeridx, channels=[542]):
        layers = model.model.layers
        for c in channels:
            for idx, layer in enumerate(layers):
                if idx in alayeridx:
                    W_out = layer.self_attn.o_proj
                    weight = W_out.weight.data
                    weight_mean = weight.mean(0)
                    weight[c] = weight_mean
                    # weight[c] = torch.zeros_like(weight_mean)
                    W_out.weight.data = weight
                if idx in mlayeridx:
                    W_out = layer.mlp.down_proj
                    weight = W_out.weight.data
                    weight_mean = weight.mean(0)
                    weight[c] = weight_mean
                    # weight[c] = torch.zeros_like(weight_mean)
                    W_out.weight.data = weight
            print(f"finish weight replace in lm selfattn oproj layers{alayeridx}- channel {c}")
            print(f"finish weight replace in lm mlp down layers{mlayeridx}- channel {c}")
    replace_parameter_CO_Woutlm(model, attn_layers, mlp_layers, channels)
    channels = [542, 121, 407, 711]
    attn_layers = vnumbers
    mlp_layers = vnumbers
    def replace_parameter_CO_Wout(model, alayeridx, mlayeridx, channels=[542]):
        for c in channels:
            layers = model.model.vision_tower.vision_tower.vision_model.encoder.layers
            for idx, layer in enumerate(layers):
                if idx in alayeridx:
                    W_out = layer.self_attn.out_proj
                    weight = W_out.weight.data
                    weight_mean = weight.mean(0)
                    weight[c] = weight_mean # maybe try mean(0)?
                    # weight[c] = torch.zeros_like(weight_mean)
                    W_out.weight.data = weight
                    bias = W_out.bias.data
                    bias_mean = bias.mean()
                    bias[c] = bias_mean
                    W_out.bias.data = bias
                if idx in mlayeridx:
                    W_out = layer.mlp.fc2
                    weight = W_out.weight.data
                    weight_mean = weight.mean(0)
                    weight[c] = weight_mean # maybe try mean(0)?
                    # weight[c] = torch.zeros_like(weight_mean)
                    W_out.weight.data = weight
                    bias = W_out.bias.data
                    bias_mean = bias.mean()
                    bias[c] = bias_mean
                    W_out.bias.data = bias
            print(f"finish weight/bias replace in selfattn oproj layers{alayeridx}- channel {c}")
            print(f"finish weight/bias replace in mlp down layers{mlayeridx}- channel {c}")
    # replace_parameter_CO_Wout(model, attn_layers, mlp_layers, channels)
    def register_custom_block_hooks(keys_to_mitigate=None):
        block_fusion = model.model
        block_vision = model.model.vision_tower.vision_tower.vision_model.encoder
        if len(tsnumbers)>0:
            block_fusion.act_restore = []
        if len(vsnumbers)>0:
            block_vision.act_restore = []
        for idx, block in enumerate(model.model.vision_tower.vision_tower.vision_model.encoder.layers):
            if idx in vnumbers:
                if len(vsnumbers)>0:
                    block.act = block_vision.act_restore
                enable_vision_custom_block(block, idx)
                print(f'set up replace in vision layer {idx}')
            if idx in vsnumbers:
                block.act = block_vision.act_restore
                enable_vision_custom_block(block, idx, mode='restore')
                print(f'set up restore in vision layer {idx}')
        for idx, block in enumerate(model.model.layers):
            if idx in tnumbers:
                if len(tsnumbers)>0:
                    block.act = block_fusion.act_restore
                enable_text_custom_block(block, idx)
                print(f'set up replace in fusion layer {idx}')
            if idx in tsnumbers:
                block.act = block_fusion.act_restore
                enable_text_custom_block(block, idx, mode='restore')
                print(f'set up restore in fusion layer {idx}')
    
    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        idx = line["id"]
        question = line['conversations'][0] # select the human
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs
        # if args.imageonly and 'image' not in line:
        #     continue
        if 'image' in line:
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        else:
            if args.imageonly:
                continue
            images = None
            image_sizes = None

        if args.single_pred_prompt:
            qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
            cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # if len(tsnumbers)>0:
        #     block_fusion = model.model
        #     block_fusion.act_restore = []
        #     print('reset restore fusion')
        # if len(vsnumbers)>0:
        #     block_vision = model.model.vision_tower.vision_tower.vision_model.encoder
        #     block_vision.act_restore = []
        #     print('reset restore vision') # useless here
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        # input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
        # register_custom_block_hooks()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_token,
                use_cache=True,
                
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        ans_file.flush()
    ans_file.close()



def replace_outlier_with_(act, key='median', threshold=20, topk = 10):
    act_ = act.flatten().abs()
    act_restore = []
    if act_.max() < 50:
        return act_restore# generate later forward stages have no massive activation
    # mean = torch.mean(act_)
    # median = torch.median(act_)
    # top_10_values, top_10_indices = torch.topk(act_, topk)
    # top_10_positions = torch.unravel_index(top_10_indices, act.shape)
    # for value, position in zip(top_10_values, zip(*top_10_positions)):
    #     if value > threshold * median and value > 50:
    #         act[position] = mean
    act_ = act.abs()
    mean = torch.mean(act_, dim=(1, 2))
    median = torch.median(act_.flatten(1), dim=-1).values
    zero = torch.zeros_like(mean)
    topk_values, topk_indices = torch.topk(act_.flatten(1), topk)
    topk_indices = torch.unravel_index(topk_indices, act.shape[1:])
    for bs, (value, indice) in enumerate(zip(topk_values, zip(*topk_indices))):
        for v, idx in zip(value, zip(*indice)):
            if v > 20:#threshold * median[bs]
                act_restore.append((act[bs][idx].clone(), idx, bs))
                if key == 'median':
                    act[bs][idx] = median[bs]
                elif key == 'mean':
                    act[bs][idx] = mean[bs]
                else:
                    act[bs][idx] = zero[bs]
    return act_restore
    # topk_values, topk_indices = torch.topk(act_.flatten(1), topk)
    # batch_size, height, width = act.shape  # Assuming `act` is a 4D tensor (B, C, H, W)
    # topk_indices = topk_indices % (height * width)
    # topk_row = topk_indices // width  # Compute the row index
    # topk_col = topk_indices % width  # Compute the column index
    # for bs, (value, indice) in enumerate(zip(topk_values, zip(topk_row, topk_col))):
    #     for v, idx in zip(value, zip(*indice)):
    #         if v > threshold * median[bs] and v > 50:
    #             if key == 'median':
    #                 act[bs][idx] = median[bs]
    #             elif key == 'mean':
    #                 act[bs][idx] = mean[bs]
    #             else:
    #                 act[bs][idx] = zero[bs]
def restore_outlier_with_(act, act_restore, key='median', threshold=300, topk = 30):
    act_ = act.abs()
    if act_.max() < 50:
        return # generate later forward stages have no massive activation, see if skip generate step
    for arestore in act_restore:
        v, idx, bs = arestore
        act[bs][idx] += v
    

def visionblock_custom_forward(
                self,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                causal_attention_mask: torch.Tensor,
                output_attentions: Optional[bool] = False,
            ) -> Tuple[torch.FloatTensor]:
        batchid=0
        if self.mode == 'replace':
            print(f"x1  before change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
            # replace_outlier_with_CO(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
            try:
                self.act += replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
                print(f"x1  after change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}, {len(self.act)}")
            except:
                replace_outlier_with_(hidden_states, 'mean', threshold=200)
                print(f"x1  after change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
            
        # replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm1 x2
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        # replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the input of norm2 y1
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm2 y2
        hidden_states = self.mlp(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the output of mlp y5
        hidden_states = residual + hidden_states
        if self.mode == 'restore':
            print(f"nextlayer x1 before restore max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
            restore_outlier_with_(hidden_states, set(self.act)) # here the input of norm1 x1
            print(f"nextlayer x1 after restore max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}, {len(set(self.act))}")
            # self.act = []
        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs



# from transformers.cache_utils import Cache
# from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
# from transformers.processing_utils import Unpack
def llamablock_custom_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        batchid = 0
        if self.mode == 'replace':
            print(f"x1  before change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
            # replace_outlier_with_CO(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
            self.act += replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the input of norm1 x1
            print(f"x1  after change max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}, {len(self.act)}")
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm1 x2
        # Self Attention
        # hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        # replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the input of norm2 y1
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean') # here the output of norm2 y2
        hidden_states = self.mlp(hidden_states)
        # replace_outlier_with_(hidden_states, 'mean', threshold=200) # here the output of mlp y7
        hidden_states = residual + hidden_states
        if self.mode == 'restore':
            print(f"nextlayer x1 before restore max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}, {len(set(self.act))}")
            restore_outlier_with_(hidden_states, set(self.act)) # here the input of norm1 x1
            # print(self.act)
            print(f"nextlayer x1 after restore max at batch{batchid}: {hidden_states[batchid].abs().max().detach()}")
            self.act = []
        outputs = (hidden_states,)


        if output_attentions:
            outputs += (self_attn_weights,)
        # if use_cache:
        #     outputs += (present_key_value,)
        return outputs

def enable_vision_custom_block(block, block_id, mode='replace'):
    block.block_id = block_id
    import types
    block.mode = mode
    block.forward = types.MethodType(visionblock_custom_forward, block)

def enable_text_custom_block(block, block_id, mode='replace'):
    block.block_id = block_id
    block.mode = mode
    # block.
    import types
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
    parser.add_argument("--max_token", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    parser.add_argument("--useflash", action="store_true")
    parser.add_argument("--imageonly", action="store_true")
    parser.add_argument('--tnumbers', type=str, default="[]", help="A Python-style list of integers")
    parser.add_argument('--vnumbers', type=str, default="[]", help="A Python-style list of integers")
    parser.add_argument('--vsnumbers', type=str, default="[]", help="A Python-style list of integers")
    parser.add_argument('--tsnumbers', type=str, default="[]", help="A Python-style list of integers")
    args = parser.parse_args()
    # global topk = args.topk
    eval_model(args)
