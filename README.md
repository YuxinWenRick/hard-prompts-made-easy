# Hard Prompts Made Easy: Discrete Prompt Tuning for Language Models

<img src=examples/teaser.png  width="70%" height="40%">

This code is the official implementation of [Hard Prompts Made Easy]().

If you have any questions, feel free to email Yuxin (<ywen@umd.edu>).

## About

From a given image, we first optimize a hard prompt using the PEZ algorithm and CLIP encoders. Then, we take the optimized prompts and feed them into Stable Diffusion to generate new images. The name PEZ (hard ***P***rompts made ***E***a**Z**y) was inspired from the [PEZ candy dispenser](https://us.pez.com/collections/dispensers).

## Try out
You can try out our demos on Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VSFps4siwASXDwhK_o29dKA9COvTnG8A?usp=sharing) or Huggingface Space [![Generic badge](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue.svg)](https://huggingface.co/spaces/tomg-group-umd/pez-dispenser).

More Jupyter notebook examples can be found in the `examples/` folder.

We recommand to run more shots to obtain more desirable prompts.

## Dependencies

- PyTorch => 1.13.0
- transformers >= 4.23.1
- diffusers >= 0.11.1
- sentence-transformers >= 2.2.2
- ftfy >= 6.1.1
- mediapy >= 1.1.2

## Usage
```python
import open_clip
from optim_utils import * 
import argparse
from PIL import Image

# load the target image
image = Image.open(image_path)

# load args
args = argparse.Namespace()
args.__dict__.update(read_json("sample_config.json"))

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)

# You may modify the hyperparamters
args.prompt_len = 8 # number of tokens for the learned prompt

# optimize prompt
learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=[image])
print(learned_prompt)
```

Note: \
```prompt_len```: number of tokens in the optimized prompt \
```batch_size```: number of target images/prompts been used for each iteraion \
```prompt_bs```: number of intializations

## Langugae Model Prompt Experiments
You may check the code in `prompt_lm/` folder.
