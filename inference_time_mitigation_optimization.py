"""
This script stores all the caption attributions in a json file in outputs/attributions/{method}.json
"""
import json
import os
import hashlib

import torch
from diffusers import DDIMScheduler
import json
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from local_sd_pipeline import LocalStableDiffusionPipeline
from optim_utils import *


@hydra.main(version_base=None, config_path="configs", config_name="inference_time_mitigation_optimization")
def main(cfg: DictConfig):

    cfg = instantiate(cfg)

    # ---------------------- #
    # (1) setup captions and attributions
    # load the json full of captions
    with open(cfg.mem_captions_path, 'r') as f:
        mem_captions = json.load(f)

    # load the json with the attributions
    image_path = os.path.join(cfg.out_dir, "inference_time_mitigation_optimization", cfg.mitigation_method.name, "images")
    prompt2image_path = os.path.join(cfg.out_dir, "inference_time_mitigation_optimization", cfg.mitigation_method.name, "prompt2image.json")
    if not os.path.exists(image_path):
        os.makedirs(image_path, exist_ok=True)
        with open(prompt2image_path, 'w') as f:
            json.dump({}, f)

    with open(prompt2image_path, 'r') as f:
        prompt2image = json.load(f)

    # ---------------------- #
    # (2) setup model
    # setup the model with the local diffusers pipeline
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe: LocalStableDiffusionPipeline = LocalStableDiffusionPipeline.from_pretrained(
        cfg.model.model_id,
        torch_dtype=torch.bfloat16,
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
        if prompt in prompt2image:
            print(f"Skipping prompt '{prompt}' as it already exists")
            continue

        prompt_hashed = hashlib.sha256(prompt.encode()).hexdigest()
        
        auged_prompt_embeds_history = pipe.aug_prompt(
            prompt,
            num_inference_steps=cfg.model.num_inference_steps,
            guidance_scale=cfg.model.guidance_scale,
            num_images_per_prompt=cfg.mitigation_method.num_images_per_prompt,
            target_steps=[cfg.mitigation_method.target_step],
            lr=cfg.mitigation_method.lr,
            optim_iters=cfg.mitigation_method.optim_steps,
            print_optim=True,
            method=cfg.mitigation_method.name,
            return_history=True
        )

        set_random_seed(cfg.seed)
        for i, auged_prompt_embeds in enumerate(auged_prompt_embeds_history):
            outputs, track_stats = pipe(
                prompt_embeds=auged_prompt_embeds,
                num_inference_steps=cfg.model.num_inference_steps,
                guidance_scale=cfg.model.guidance_scale,
                num_images_per_prompt=cfg.model.num_images_per_prompt,
                track_noise_norm=True,
                height=cfg.model.image_size,
                width=cfg.model.image_size,
            )
            pil_images = outputs.images
            image_path_with_hash = os.path.join(image_path, f"{prompt_hashed}")
            os.makedirs(image_path_with_hash, exist_ok=True)
            for j, pil_image in enumerate(pil_images):
                pil_image.save(os.path.join(image_path_with_hash, f"{i:04d}_{j:02d}.png"))

            
        prompt2image[prompt] = prompt_hashed

        # save the attributions
        with open(prompt2image_path, 'w') as f:
            json.dump(prompt2image, f, indent=4)
            

if __name__ == "__main__":
    main()