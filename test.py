from open_clip.tokenizer import get_pairs
def test_get_pairs():
    # Test case 1: Word with no repeating characters
    word = ('h', 'e', 'l', 'o')
    expected_pairs = {('h', 'e'), ('e', 'l'), ('l', 'o')}
    assert get_pairs(word) == expected_pairs
    
    # Test case 2: Word with repeating characters
    word = ('h', 'e', 'l', 'l', 'o')
    expected_pairs = {('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o')}
    assert get_pairs(word) == expected_pairs
    
    # Test case 3: Single-character word
    word = ('a',)
    expected_pairs = set()
    assert get_pairs(word) == expected_pairs
