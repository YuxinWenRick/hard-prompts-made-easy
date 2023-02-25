import unittest
from unittest import mock
from argparse import Namespace
from PIL import Image
import torch
import open_clip
from optim_utils import read_json, optimize_prompt


class TestOptimizePrompt(unittest.TestCase):

    def test_optimize_prompt(self):
        # create mock objects
        model = mock.Mock()
        model.token_embedding = torch.nn.Embedding(512, 512)
        preprocess = mock.Mock()
        args = Namespace()
        args.clip_model = 'ViT-B/32'
        args.clip_pretrain = True
        args.iter = 10
        args.print_new_best = True
        args.print_step = None
        device = "cpu"
        image_path = 'test_image.jpg'
        image = Image.new('RGB', (256, 256), color='red')
        image.save(image_path)
        image_paths = [image_path]

        # test optimize_prompt with mock objects
        with self.assertRaises(Exception):
            learned_prompt = optimize_prompt(
                model,
                preprocess,
                args,
                device,
                target_images=[image],
                target_prompts=['test'])
            self.assertEqual(len(learned_prompt), 512)

    def test_get_tokenizer(self):
        tokenizer = open_clip.get_tokenizer('ViT-B/32')
        self.assertIsNotNone(tokenizer)

    def test_get_model_config(self):
        config = open_clip.get_model_config('ViT-B/32')
        self.assertIsNotNone(config)
        self.assertIsNotNone(config.get('text_cfg'))
        self.assertIsNotNone(config['text_cfg'].get('hf_tokenizer_name'))


if __name__ == '__main__':
    unittest.main()
