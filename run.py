import sys
from PIL import Image

if len(sys.argv) < 2:
  sys.exit("""Usage: python run.py path-to-image [path-to-image-2 ...]
Passing multiple images will optimize a single prompt across all passed images, useful for style transfer.
""")

config_path = "sample_config.json"

image_paths = sys.argv[1:]

# load the target image
images = [Image.open(image_path) for image_path in image_paths]

# defer loading other stuff until we confirm the images loaded
import argparse
import open_clip
from optim_utils import *

print("Initializing...")

# load args
args = argparse.Namespace()
args.__dict__.update(read_json(config_path))

# You may modify the hyperparamters here
args.print_new_best = True

# load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(args.clip_model, pretrained=args.clip_pretrain, device=device)

print(f"Running for {args.iter} steps.")
if getattr(args, 'print_new_best', False) and args.print_step is not None:
  print(f"Intermediate results will be printed every {args.print_step} steps.")

# optimize prompt
learned_prompt = optimize_prompt(model, preprocess, args, device, target_images=images)
print(learned_prompt)
