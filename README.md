# Detecting, Explaining, and Mitigating Memorization in Diffusion Models ( + geometric functionalities)

This is a fork of the original repository, [Detecting, Explaining, and Mitigating Memorization in Diffusion Models](https://github.com/YuxinWenRick/diffusion_memorization). We have included some local intrinsic dimension (LID) functionalities for mitigating memorization to this repo. We use an LID estimator from the paper [A Geometric View of Data Complexity](https://arxiv.org/abs/2406.03537) which proposes a method that is differentiable by design and use that as a replacement for the classifier free oprimization (CFG) norm that was originally proposed in the paper. We adapt FLIPD for diffusion denoising implicit models (DDIM) that has been used to run the experiments in the paper. We also include pure score norm oprimization and see how it compares to the other two methods.

In addition to that, we have also included a GPT-based method for prompt perturbation using token attributions obtained from these methods. For the base functionalities, please consult the original repository.

**Remark 1**: For all the command-line scripts here, you have control over the configuration and hyperparameters with [hydra](https://hydra.cc/), please check `configs/` for more details.

**Remark 2**: All the logs in this project will be stored in the `outputs` directory. There are also some default logs stored in the repo to get to know the structure of the logs.

## Setting up environment

Run the following script to set up the environment:

```bash
conda env create -f env.yml
conda activate geometric_diffusion_memorization
```

## Prompt attribution

When running the following script, it will take prompts that are stored in the `match_verbatim_captions.json` and run attribution methods on them. Attribution methods basically take a caption and see how influential a token is for memorization. We have three attribution methods, `cfg_norm`, `score_norm`, and `flipd`. The first one is what was proposed in the original paper, the second one is a simple score norm, and the third one is the LID-based method.

```bash
# get the token attributions for a set of memorized prompts
python store_attributions.py attribution_method=<score_norm|cfg_norm|flipd>
```

This will store a json file in `outputs/attributions` that contains all the token attributions obtained via any of the metrics above! Note that you can also specify your own set of prompts.

### Perturbation

After obtaining these attributions, we can use that information to perturb the prompts. For that, we will use the OpenAI API to generate new relevant prompts that do not contain these sensitive tokens that are causing memorization.
Prompt perturbation is done through our `perturb_gpt.py` script. Before that, you have to store your `OPENAI_API_KEY` in a `.env` file in the root directory of this repo, meaning, you should have a file `.env` with the following content:

```bash
dotenv set OPENAI_API_KEY '<your-key>'
```

You can then run the following script:

```bash
python perturb_gpt.py attribution_method=<score_norm|cfg_norm|flipd|random>
```

This script looks at the already stored attributions in `outputs/attributions`, therefore, make sure to run the last part first before running this script -- we have however included some default attributions in the repo. Note that we also have a `random` method that perturbs the prompts randomly. Finally, after running this script all the perturbed prompts will be stored in a separate file in `outputs/perturbed_prompts`.

# Notebooks

Please check out [this](examples/inference_time_mitigation.ipynb) example notebook to see how you can use our geometric method for mitigating memorization in a diffusion model.
