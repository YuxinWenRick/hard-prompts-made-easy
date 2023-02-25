import unittest
import argparse

import torch
from PIL import Image
from optim_utils import optimize_prompt

class TestOptimizePrompt(unittest.TestCase):
    
    def test_optimize_prompt(self):
        # initialize args
        args = argparse.Namespace()
        args.clip_model = 'ViT-B/32'
        args.clip_pretrain = True
        args.image_size = 224
        args.init_lr = 0.1
        args.iter = 200
        args.beta1 = 0.9
        args.beta2 = 0.999
        args.epsilon = 1e-08
        args.weight_decay = 0.0
        args.num_iterations = 1
        args.blur_every = 4
        args.grad_clip_norm = 0.1
        args.center_bias = False
        args.mode = 'keep'
        args.seed = 42
        
        # load test image
        image_path = 'test-image.jpeg'
        target_image = Image.open(image_path)
        
        # load CLIP model
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            args.clip_model, pretrained=args.clip_pretrain, device=DEVICE)
        
        # run optimization
        learned_prompt = optimize_prompt(
            model,
            preprocess,
            args,
            DEVICE,
            target_images=[target_image])
        
        # assert result is a string
        self.assertIsInstance(learned_prompt, str)
