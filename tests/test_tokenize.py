from utils.tokenize import tokenize 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_tokenize_accepts_string(): 
    tokens = tokenize("kvakk, dette er en streng")
    assert isinstance(tokens, list)
    assert isinstance(tokens[0], str)
    

def test_tokenize(): 
    tokens = (tokenize("The quick brown fox jumped over the lazy dog"))
    assert "fox" in tokens
    assert "quick" in tokens 


def test_removes_stop_words(): 
    sentence = "This is a sentence, it has several words"
    tokens = tokenize(sentence)
    expected = ["sentence", "words"]
    assert tokens == expected, print(tokens, expected)
     

def test_does_not_remove_non_stop_words(): 
    sentence = "Marcus Aurelius Roman Emperor"
    
    tokens = tokenize(sentence)
    for word in ["marcus", "aurelius", "roman", "emperor"]:
        assert word in tokens, print(tokens)
    
    
def test_ignores_non_alpha(): 
    sentence = "52 B-52s are stationed in Norway"
    tokens = tokenize(sentence)
    assert "52" not in tokens 
    assert "norway" in tokens 