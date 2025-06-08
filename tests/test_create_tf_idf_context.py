import math
import pytest 
from src.most_similar import create_tf_idf_context 
from utils.tokenize import tokenize
import numpy as np 





test_corpus = [
    "This sci-fi film was groundbreaking and changed the genre forever. You'll be sitting at the edge of your seat",
    "An emotional, thrilling journey that explores deep themes with stunning visuals.",
    "A thrilling adventure that keeps you on the edge of your seat.",
]

tokens = [word for document in test_corpus for word in tokenize(document)]
all_words_global = set(tokens)


def test_all_words_generated_as_expected(): 
    context = create_tf_idf_context(test_corpus)
    assert all_words_global.issubset(context[0])
    
def test_check_return_types(): 
    all_words, term_index_dict, idf_vector = create_tf_idf_context(test_corpus)
    assert isinstance(all_words, set)
    assert isinstance(next(iter(all_words)), str)  # safer than pop()
    assert isinstance(term_index_dict, dict)
    assert isinstance(term_index_dict["groundbreaking"], int) 
    assert isinstance(idf_vector, np.ndarray)
    assert isinstance(idf_vector[0], np.float64), print(type(idf_vector[0]))
    
    # test sorting, just for fun. 
    assert sorted(list(all_words)) == sorted(list(term_index_dict.keys()))


def test_mapping(): 
    # extra test for mapping
    
    local_test_corpus = [
        "Aaron, angst, along", 
        "Bravado Benedict Cumberbatch", 
        "Zoro the leader of men"
    ]
    
    all_words, mapping_dict, _ = create_tf_idf_context(local_test_corpus)
    sorted_words = sorted(all_words)

    for idx, word in enumerate(sorted_words):
        assert mapping_dict[word] == idx, f"Word '{word}' should map to {idx}, got {mapping_dict[word]}"
    
    print(mapping_dict.items())


def test_idf_vector(): 
    local_corpus = [
        "This is a frog", 
        "This is a log", 
        "This is a song",
        "This is a frog"
    ]
    
    all_words, term_index_mapping, idf_vector = create_tf_idf_context(local_corpus)
    
    N = len(local_corpus)
    
    frog_index = term_index_mapping["frog"]
    expected_frog_idf = math.log(N / (1 + 2))
    assert math.isclose(expected_frog_idf, 0.287682072, rel_tol=1e-5)
    assert math.isclose(idf_vector[frog_index], 0.287682072, rel_tol=1e-5) 
    
    log_index = term_index_mapping["log"]
    expected_log_idf = math.log(N / (1 + 1))
    
    assert abs(expected_log_idf - idf_vector[log_index] ) < 1e-6
    
    song_index = term_index_mapping["song"]
    expected_song_idf = math.log(N / (1 + 1))
    
    assert abs(expected_song_idf - idf_vector[song_index]) < 1e-6
    
    
def test_negative_idfs(): 
    
    local_corpus = [ 
                    "I have a sword", 
                    "You have a sword", 
                    "This is a sword"]
    
    all_words, term_index_mapping, idf_vector = create_tf_idf_context(local_corpus)
    
    # all words should only contain sword, which is in all docs
    
    N = len(local_corpus)
    
    expected_idf = math.log(N / (1 + 3))
    # should be negative value
    assert expected_idf < 0
    
    assert abs(expected_idf - -0.287682072) < 1e-6
    
    
def test_finiteness(): 
    _, _, idfvector = create_tf_idf_context(test_corpus)
    
    assert np.all(np.isfinite(idfvector))