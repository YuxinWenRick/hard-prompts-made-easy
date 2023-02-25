import os
import pytest
from PIL import Image
from run import optimize_image

# Set the path to the test image
IMAGE_PATH = "test-image.jpeg"

def test_optimize_image():
    # Load the test image
    image = Image.open(IMAGE_PATH)

    # Optimize the prompt for the image
    learned_prompt = optimize_image(image)

    # Assert that the learned prompt is not empty
    assert learned_prompt

    # Assert that the learned prompt is a string
    assert isinstance(learned_prompt, str)

    # Assert that the learned prompt contains no whitespace
    assert not any(char.isspace() for char in learned_prompt)
