from open_clip.tokenizer import tokenize
import torch

def test_tokenize():
    texts = ["This is a test string 1", "This is a test string 2"]
    expected_output = torch.tensor([
        [49408,   250,    11,  1329,   830,     7,  7208,    88,  1823,   443, 49407,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
        [49408,   250,    11,  1329,   830,     7,  7208,    88,  1823,   443, 49407,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0,     0]
    ])
    output = tokenize(texts)
    assert torch.equal(output, expected_output)
