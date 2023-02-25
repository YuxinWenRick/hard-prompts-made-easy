from open_clip.tokenizer import tokenize
import torch

texts = ["This is the first text to tokenize", 
         "This is the second text to tokenize, which is a bit longer than the first one"]
context_length = 77

# Call the function
tokens = tokenize(texts, context_length)

# Check the output shape
assert tokens.shape == (2, context_length)

# Check the output values for the first text
expected_output_1 = torch.tensor([49408, 250, 11, 1329, 830, 7, 7208, 88, 1823, 443, 250, 25852, 
                                  172, 719, 250, 11, 39539, 6085, 4025, 481, 728, 49409, 0, 0, 
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
assert torch.equal(tokens[0], expected_output_1)

# Check the output values for the second text
expected_output_2 = torch.tensor([49408, 250, 11, 1329, 830, 7, 7208, 88, 1823, 443, 250, 25852, 
                                  172, 719, 250, 11, 1329, 830, 7, 1279, 66, 830, 16, 11402, 
                                  11, 443, 250, 22, 2057, 54, 419, 7145, 1085, 25852, 172, 
                                  719, 250, 11, 2019, 8, 33, 54, 4444, 5384, 49409, 0, 0, 0, 
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
assert torch.equal(tokens[1], expected_output_2)
