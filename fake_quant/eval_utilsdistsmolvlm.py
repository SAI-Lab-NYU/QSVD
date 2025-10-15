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
def evaluator(model, testenc, dev, args, image_processor):
    from vlmeval.inference import infer_data
    from vlmeval.smp import dump, get_rank_and_world_size, string, load
    from vlmeval.dataset import DATASET_TYPE
    import pandas as pd
    rank, world_size = get_rank_and_world_size()
    # if world_size > 1:
    #     local_rank = os.environ.get('LOCAL_RANK', 0)
    #     torch.cuda.set_device(int(local_rank))
    #     dist.init_process_group(
    #         backend='nccl',
    #         timeout=datetime.timedelta(seconds=int(os.environ.get('DIST_TIMEOUT', 3600)))
    #     )
    _, testenc = testenc
    # torch.cuda.synchronize()
    logging.info("start moving model to dev")
    model = model.to(rank)
    logging.info("moving model to dev")

    if world_size > 1:
        dist.barrier()
    def build_prompt_llava(line, dataset):
        # llamav use dataset prompt
        pass
    def ensure_image_url(image: str) -> str:
        prefixes = ['http://', 'https://', 'file://', 'data:image;']
        if any(image.startswith(prefix) for prefix in prefixes):
            return image
        if os.path.exists(image):
            return 'file://' + image
        raise ValueError(f'Invalid image: {image}')

    def message_to_prompt(message, image_processor, model):
        from transformers.image_utils import load_image
        from PIL import Image
        replace_mapping = {
            "\nOptions:": "\nChoices:",
            "Please select the correct answer from the options above.": "Answer with the letter.",
        }
        prompt, images = "<|im_start|>User:", []
        for msg in message:
            if msg["type"] == "image":
                img = load_image(msg["value"])
                images.append(img)
                prompt += "<image>"
            elif msg["type"] == "text":
                instruction = msg["value"].strip()
                for k, v in replace_mapping.items():
                    instruction = instruction.replace(k, v)
                prompt += instruction
        prompt += "<end_of_utterance>\nAssistant: Answer:"
        images = (
            [images]
            if isinstance(images, Image.Image)
            else images
        )
        inputs = image_processor(
            text=prompt, images=images, return_tensors="pt"
        ).to("cuda")
        return inputs

    
    def infer_data_llava(lm, args, verbose=False):

        dataset = testenc
        dataset_name = dataset.dataset_name
        rank, world_size = get_rank_and_world_size()
        # if rank == 0:
        logging.info("Start selecting split data!")
        # Each rank writes to a unique output file
        out_file = args.save_path + f'/vlm_eval_rank{rank}.pkl'

        sheet_indices = list(range(rank, len(dataset), world_size))
        data = dataset.data.iloc[sheet_indices]
        data_indices = [i for i in data['index']]
        res = {}

        if os.path.exists(out_file):
            res.update(load(out_file))

        if world_size > 1:
            dist.barrier()
        # Check if all results are already computed
        all_finished = all(idx in res for idx in data['index'])


        data = data[~data['index'].isin(res)]
        if world_size > 1:
            dist.barrier()
        
        logging.info("finish selecting split data!")
        for i in tqdm(range(len(data))):
            idx = data.iloc[i]['index']
            # message = build_prompt_llava(data.iloc[i], dataset) # 
            # inputs_id, image, stopping_criteria = message_to_prompt(
            #     message, image_processor, model, tokenizer
            # )
            message = dataset.build_prompt(data.iloc[i])
            inputs = message_to_prompt(message, image_processor, model)
            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs, 
                    max_new_tokens=16, 
                    use_cache=True)
            torch.cuda.empty_cache()
            out = image_processor.batch_decode(
                generated_ids[:, inputs["input_ids"].size(1) :], skip_special_tokens=True
                )[0]
            response = out.strip()
            # if verbose:
            #     print(response, flush=True)

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
