"""
This script perturbs text tokens based on GPT.
"""

import json
import os

import torch
import json
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import dotenv

from local_sd_pipeline import LocalStableDiffusionPipeline
from optim_utils import *
from openai import OpenAI

with open('match_verbatim_captions.json') as f:
    all_mem_captions = json.load(f)
    

GPT_INSTRUCTION_FSTR = """
I have the following caption as a sequence of tokens:
{original_tokens}
I want to create a new caption based on this one, but I want to perturb the following tokens:
{target_tokens}
These are the rules to follow for perturbing tokens:
1. If the token is a special character or punctuation without significant semantics, you can remove it or change it to any special character
2. If the token is a number, you can replace it with another number that is close to it
3. If the token is a special name, such as the name of someone or some place or some culture, it should not be replaced
4. If the token is any other word, you can replace and rephrase it with any synonym that makes sense in the context
Given these requirements, please provide me with a new caption, not as a sequence of tokens, but as a natural language sentence that sematically matches closely with the original caption except for the perturbed tokens. Do not say anything else in response, only provide me with the new caption.
"""

def perturb_prompt(prompt_tokens: list, tokens_and_attributes: list, level: int, method: str = 'flipd', temperature: float = 0.001):
    """
    Takes a list of prompt tokens, the number of tokens to change (level), tokens and attributes as a list [[tok, attr], ...]
    and method (which is either random or looks at the token attributes)
    """
    dotenv.load_dotenv(override=True)
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    sorted_list = sorted(tokens_and_attributes, key=lambda x: x[1], reverse=True)
    tok_sorted_by_attribute = [tok for tok, _ in sorted_list]
    logits = np.array([attr/ temperature for _, attr in sorted_list])
    prob_values = np.exp(logits) / np.sum(np.exp(logits))
    if method == 'random':
        prob_values = np.ones_like(prob_values) / len(prob_values)
    choose_level = min(level, len(prob_values))
    selected_tokens = np.random.choice(len(prompt_tokens), choose_level, replace=False, p=prob_values)
    selected_tokens = [tok_sorted_by_attribute[i] for i in selected_tokens]

    gpt_instruction = GPT_INSTRUCTION_FSTR.format(
        original_tokens=" ".join(prompt_tokens),
        target_tokens=" ".join(selected_tokens)
    )
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": gpt_instruction},
        ],
        max_tokens=50,
    )
    new_prompt = completion.choices[0].message.content
    return new_prompt, selected_tokens

@hydra.main(version_base=None, config_path="configs", config_name="perturb_gpt")
def main(cfg: DictConfig):
    
    cfg = instantiate(cfg)

    # ---------------------- #
    # (1) load captions and attributions
    # load the json full of captions
    attribution_method_ = cfg.attribution_method.name if cfg.attribution_method.name != 'random' else 'cfg_norm'
    attribution_path = os.path.join(cfg.out_dir, "attributions", f"{attribution_method_}.json")
    with open(attribution_path, 'r') as f:
        attributions = json.load(f)

    gpt_perturbations = os.path.join(cfg.out_dir, "attributions", f"{cfg.attribution_method.name}-perturbations.json")
    if not os.path.exists(gpt_perturbations):
        os.makedirs(os.path.dirname(gpt_perturbations), exist_ok=True)
        with open(gpt_perturbations, 'w') as f:
            json.dump({}, f)
    with open(gpt_perturbations, 'r') as f:
        perturbations = json.load(f)    

    # ---------------------- #
    # (2) setup pipeline tokenizer
    pipe = LocalStableDiffusionPipeline.from_pretrained(
        cfg.model.model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )

    # ---------------------- #
    # (3) iterate over all the captions and find attributions
    for prompt, tok_attr in attributions.items():
        if prompt in perturbations:
            print(f"Skipping prompt '{prompt}' as it already exists")
            continue

        prompt_tokens = pipe.tokenizer.encode(prompt)
        prompt_tokens = prompt_tokens[1:-1]
        prompt_tokens = prompt_tokens[:75]
        prompt_tokens = [pipe.tokenizer.decode(tok) for tok in prompt_tokens]
        
        if prompt not in perturbations:
            perturbations[prompt] = {}

        for level in cfg.levels:

            perturbations[prompt][level] = perturbations.get(level, [])
            for _ in range(cfg.num_perturbations - len(perturbations[prompt][level])):
                
                # tokenize and all
                new_prompt, selected_tokens = perturb_prompt(
                    prompt_tokens=prompt_tokens,
                    tokens_and_attributes=tok_attr,
                    level=level,
                    method=cfg.attribution_method.name,
                )
                perturbations[prompt][level].append(
                    {
                        "perturbed_prompt": new_prompt,
                        "selected_tokens": selected_tokens
                    }
                )
                
        with open(gpt_perturbations, 'w') as f:
            json.dump(perturbations, f, indent=4)
            
        print(f"Finished prompt '{prompt}'")   

if __name__ == "__main__":
    main()