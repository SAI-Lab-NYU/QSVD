import utils
import model_utils
import quant_utils
import torch
import os
import logging
from tqdm import tqdm
from vlmeval.smp import *
import torch.distributed as dist
import datetime
@torch.no_grad()
def evaluator(model, testenc, dev, args, tokenizer, image_processor):
    from vlmeval.inference import infer_data
    from vlmeval.smp import dump, get_rank_and_world_size, string, load
    from vlmeval.dataset import DATASET_TYPE, build_dataset
    import pandas as pd
    rank, world_size = get_rank_and_world_size()
    # if world_size > 1:
    #     local_rank = os.environ.get('LOCAL_RANK', 0)
    #     torch.cuda.set_device(int(local_rank))
    #     dist.init_process_group(
    #         backend='nccl',
    #         timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
    #     )
    testenc = build_dataset('SEEDBench_IMG')
    # torch.cuda.synchronize()
    model = model.to(rank)
    logging.info("moving model to dev")


    def build_prompt_llava(line, dataset):
        tgt_path = dataset.dump_image(line)
        question = line['question']
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question 
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        question += "\nAnswer with the option's letter from the given choices directly."
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=question))
        return message

    def message_to_prompt(data, image_processor, model, tokenizer):
        from llava.mm_utils import (
            process_images,
            tokenizer_image_token,
            KeywordsStoppingCriteria,
        )
        from llava.constants import IMAGE_TOKEN_INDEX
        from PIL import Image
        from abc import abstractproperty

        system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )

        def concat_tilist(message):
            text, images = "", []
            for item in message:
                if item["type"] == "text":
                    text += item["value"]
                elif item["type"] == "image":
                    text += " <image> "
                    images.append(item["value"])
            return text, images

        content, images = concat_tilist(data)
        images = [Image.open(s).convert("RGB") for s in images]
        args = abstractproperty()
        args.image_aspect_ratio = "pad"
        image_tensor = process_images(images, image_processor, args).to(
            "cuda", dtype=torch.float16
        )

        prompt = system_prompt + "USER: " + content + " ASSISTANT: "
        # ADD conv templets
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
        stopping_criteria = KeywordsStoppingCriteria(
            ["</s>"], tokenizer, input_ids
        )
        return input_ids, image_tensor, stopping_criteria

    
    def infer_data_llava(lm, args, verbose=False):

        dataset = testenc
        dataset_name = dataset.dataset_name
        rank, world_size = get_rank_and_world_size()
        if rank == 0:
            logging.info("Start selecting split data!")
        # Each rank writes to a unique output file
        out_file = args.save_path + f'/vlm_eval_rank{rank}.pkl'

        sheet_indices = list(range(rank, len(dataset), world_size))
        data = dataset.data.iloc[sheet_indices]
        data_indices = [i for i in data['index']]
        res = {}

        if os.path.exists(out_file):
            res.update(load(out_file))

        # Check if all results are already computed
        all_finished = all(idx in res for idx in data['index'])
        # if all_finished:
        #     dump({k: res[k] for k in data_indices}, out_file)
        #     return

        data = data[~data['index'].isin(res)]
        if world_size > 1:
            dist.barrier()
        
        logging.info("finish selecting split data!")
        for i in tqdm(range(len(data))):
            idx = data.iloc[i]['index']
            message = build_prompt_llava(data.iloc[i], dataset)
            inputs_id, image, stopping_criteria = message_to_prompt(
                message, image_processor, model, tokenizer
            )

            with torch.inference_mode():
                if not args.dosample:
                    response = model.generate(
                        inputs_id, images=image,
                        do_sample=False,
                        max_new_tokens=16,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                    )
                else:
                    response = model.generate(
                        inputs_id, images=image,
                        do_sample=True, temperature=0.1,
                        max_new_tokens=16, 
                        use_cache=True,
                        stopping_criteria=[stopping_criteria],
                    )
            torch.cuda.empty_cache()
            response = tokenizer.batch_decode(response, skip_special_tokens=True)[0].strip()
            if verbose:
                print(response, flush=True)

            res[idx] = response
            if (i + 1) % 10 == 0:
                dump(res, out_file)

        dump({k: res[k] for k in data_indices}, out_file)
        if world_size > 1:
            dist.barrier()
        # Merge results on rank 0 after inference
        if rank == 0:
            merged_results = {}
            for r in range(world_size):
                rank_file = args.save_path + f'/vlm_eval_rank{r}.pkl'
                if os.path.exists(rank_file):
                    merged_results.update(load(rank_file))

            # Dump merged results
            merged_out_file = args.save_path + '/vlm_eval.pkl'
            dump(merged_results, merged_out_file)

            # Clean up per-GPU files
            for r in range(world_size):
                rank_file = args.save_path + f'/vlm_eval_rank{r}.pkl'
                if os.path.exists(rank_file):
                    os.remove(rank_file)

            logging.info("Merged and cleaned up all per-GPU outputs.")

        return
    if world_size > 1:
        dist.barrier()
    # Run inference
    infer_data_llava(model, args, True)

    # Rank 0 handles evaluation
    rank, _ = get_rank_and_world_size()
    if rank == 0:
        logging.info("Start evaluation!")
        result_file = args.save_path + '/vlm_result.xlsx'
        data_all = load(args.save_path + '/vlm_eval.pkl')
        dataset = testenc.data
        
        # Ensure all indices are covered
        for x in dataset['index']:
            assert x in data_all

        dataset['prediction'] = [str(data_all[x]) for x in dataset['index']]
        if 'image' in dataset:
            dataset.pop('image')

        dump(dataset, result_file)

        judge_kwargs = {'nproc': 4, 'verbose': True, 'retry': 3}
        eval_results = testenc.evaluate(result_file, **judge_kwargs)
        import json
        logging.info(eval_results)
        if isinstance(eval_results, dict):
            logging.info('\n' + json.dumps(eval_results, indent=4))
        elif isinstance(eval_results, pd.DataFrame):
            if len(eval_results) < len(eval_results.columns):
                eval_results = eval_results.T
            logging.info('\n' + tabulate(eval_results))
        return eval_results
