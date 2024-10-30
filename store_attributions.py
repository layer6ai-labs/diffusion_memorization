"""
This script stores all the caption attributions in a json file in outputs/attributions/{method}.json
"""
import json
import os

import torch
from diffusers import DDIMScheduler
import json
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from local_sd_pipeline import LocalStableDiffusionPipeline
from optim_utils import *


with open('match_verbatim_captions.json') as f:
    all_mem_captions = json.load(f)
    


@hydra.main(version_base=None, config_path="configs", config_name="store_attributions")
def main(cfg: DictConfig):
    
    cfg = instantiate(cfg)

    # ---------------------- #
    # (1) setup captions and attributions
    # load the json full of captions
    with open(cfg.mem_captions_path, 'r') as f:
        mem_captions = json.load(f)

    # load the json with the attributions
    attribution_path = os.path.join(cfg.out_dir, "attributions", f"{cfg.attribution_method.name}.json")
    if not os.path.exists(attribution_path):
        os.makedirs(os.path.dirname(attribution_path), exist_ok=True)
        with open(attribution_path, 'w') as f:
            json.dump({}, f)
    with open(attribution_path, 'r') as f:
        attributions = json.load(f)

    # ---------------------- #
    # (2) setup model
    # setup the model with the local diffusers pipeline
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = LocalStableDiffusionPipeline.from_pretrained(
        cfg.model.model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # ---------------------- #
    # (3) iterate over all the captions and find attributions
    for prompt_idx, prompt in enumerate(mem_captions):
        if not cfg.prompt_start_idx <= prompt_idx < cfg.prompt_end_idx:
            continue
        if prompt in attributions:
            print(f"Skipping prompt '{prompt}' as it already exists")
            continue

        set_random_seed(cfg.seed)
        token_grads = pipe.get_text_cond_grad(
            prompt,
            num_inference_steps=cfg.attribution_method.num_inference_steps,
            guidance_scale=cfg.model.guidance_scale,
            num_images_per_prompt=cfg.attribution_method.num_images_per_prompt,
            target_steps=cfg.attribution_method.target_steps,
            method=cfg.attribution_method.name,
        )
        torch.cuda.empty_cache()

        prompt_tokens = pipe.tokenizer.encode(prompt)
        prompt_tokens = prompt_tokens[1:-1]
        prompt_tokens = prompt_tokens[:75]
        token_grads = token_grads[1:(1+len(prompt_tokens))]
        token_grads = token_grads.cpu().tolist()

        all_tokes = []

        for curr_token in prompt_tokens:
            all_tokes.append(pipe.tokenizer.decode(curr_token))

        attributions[prompt] = [(tok, grad) for tok, grad in zip(all_tokes, token_grads)]

        # save the attributions
        with open(attribution_path, 'w') as f:
            json.dump(attributions, f, indent=4)
            

if __name__ == "__main__":
    main()