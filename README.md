# Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery

<img src=examples/teaser.png  width="70%" height="40%">

This code is the official implementation of [Hard Prompts Made Easy](https://arxiv.org/abs/2302.03668).

If you have any questions, feel free to email Yuxin (<ywen@umd.edu>).

## About

From a given image, we first optimize a hard prompt using the PEZ algorithm and CLIP encoders. Then, we take the optimized prompts and feed them into Stable Diffusion to generate new images. The name PEZ (hard ***P***rompts made ***E***a**Z**y) was inspired from the [PEZ candy dispenser](https://us.pez.com/collections/dispensers).

## Try out
You can try out our demos on Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VSFps4siwASXDwhK_o29dKA9COvTnG8A?usp=sharing) or Hugging Face Space [![Generic badge](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/tomg-group-umd/pez-dispenser).

More Jupyter notebook examples can be found in the `examples/` folder.

We recommand to run more shots to obtain more desirable prompts.

## Dependencies

- PyTorch => 1.13.0
- transformers >= 4.23.1
- diffusers >= 0.11.1
- sentence-transformers >= 2.2.2
- ftfy >= 6.1.1
- mediapy >= 1.1.2

## Setup

Ensure you have python 3 installed.

Create a virtual environment, activate it, and install dependencies:
```sh
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

## Usage

A script is provided to perform prompt inversion (finding a prompt from an image or set of images). For examples of other usages, see [the examples folder](./examples).

```sh
python run.py image.png
```

You can pass multiple images to optimize a prompt across all images.

## Parameters

Config can be loaded from a JSON file. A sample config is provided at [./sample-config.json](sample_config.json).

Config has the following parameters:

- `prompt_len`: the number of tokens in the optimized prompt. 16 empirically results in the most generalizable performance. more is not necessarily better.
- `iter`: the total number of iterations to run for.
- `lr`: the learning weight for the [optimizer](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html).
- `weight_decay`: the weight decay for the optimizer.
- `prompt_bs`: number of initializations.
- `batch_size`: number of target images/prompts used for each iteration.
- `clip_model`: the name of the CLiP model for use with . `"ViT-H-14"` is the model used in SD 2.0 and Midjourney. `"ViT-L-14"` is the model used in SD 1.5. This should ideally match your target generator.
- `clip_pretrain`: the name of the pretrained model for [open_clip](https://github.com/mlfoundations/open_clip). For `"ViT-H-14"` use `"laion2b_s32b_b79k"`. For `"ViT-L-14"` use `"openai"`.
- `print_step`: if not null, how often (in steps) to print a line giving current status.
- `print_new_best`: whether to print out new best prompts whenver found. will be quite noisy initially.

## Language Model Prompt Experiments
You may check the code in `prompt_lm/` folder.
