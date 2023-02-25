import torch
from open_clip.tokenizer import tokenize
def test_tokenize():
    expected_result = torch.tensor([[49408, 364, 16, 939, 4054, 4, 175, 49409, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],[49408, 364, 16, 939, 4054, 4, 175, 49409, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    result = tokenize(["A photo of a family of ducks walking down the street"])
    assert torch.equal(expected_result, result), f"Expected {expected_result}, but got {result}"
