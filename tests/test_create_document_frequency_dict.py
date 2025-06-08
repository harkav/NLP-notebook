import pytest
from src.most_similar import create_document_frequency_dict
from utils.tokenize import tokenize

test_corpus = [
    "This sci-fi film was groundbreaking and changed the genre forever. You'll be sitting at the edge of your seat",
    "An emotional, thrilling journey that explores deep themes with stunning visuals.",
    "A thrilling adventure that keeps you on the edge of your seat.",
]

tokens = [word for document in test_corpus for word in tokenize(document)]
all_words = set(tokens)


def test_accepts_corpus():
    create_document_frequency_dict(all_words, test_corpus)
    
def test_works_with_a_single_doc():
    create_document_frequency_dict(all_words, [test_corpus[0]])
    
def test_counts_correctly_and_returns_dict(): 
    df = create_document_frequency_dict(all_words, test_corpus)
    assert isinstance(df, dict)
    assert isinstance(df["seat"], int)
    assert df.get("seat") == 2
    assert df.get("Marcus") is None
    assert df.get("thrilling") == 2
    assert df.get("explores") == 1
        