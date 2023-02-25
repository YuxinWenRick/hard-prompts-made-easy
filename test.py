def test_tokenize():
    texts = ["This is a test.", "This is another test."]
    expected_result = torch.tensor([[49408,   364,    16,   939,  4054,     4,   175, 49409,     0, ... 0],
                                    [49408,   364,    16,   939,  4054,    38,   308,  4413,  4054, ... 0]])
    assert torch.allclose(tokenize(texts), expected_result)
