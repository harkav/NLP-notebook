from collections import Counter
from src.most_similar import create_object_new, create_tf_idf_context
import pytest 
import math

from utils.tokenize import tokenize 



def test_tf_idf_value():
    corpus = ["apple apple banana", "banana orange", "apple banana"]
    input_doc = "apple banana"

    all_words, term_index_dict, idf_vector = create_tf_idf_context(corpus)

    doc_obj = create_object_new(input_doc, all_words, term_index_dict, idf_vector)

    tf_counter = Counter(tokenize(input_doc))
    N = len(corpus)

    for word, tf in tf_counter.items():
        idx = term_index_dict[word]
        expected_idf = math.log(N / (1 + sum(word in tokenize(doc) for doc in corpus)))
        expected_tfidf = tf * expected_idf
        assert math.isclose(doc_obj.vector[idx], expected_tfidf, rel_tol=1e-6), (
            f"{word} — expected {expected_tfidf}, got {doc_obj.vector[idx]}"
        )


def test_zero_tf_idf_for_absent_words():
    corpus = ["alpha beta", "beta gamma"]
    input_doc = "alpha"
    all_words, term_index_dict, idf_vector = create_tf_idf_context(corpus)
    doc_obj = create_object_new(input_doc, all_words, term_index_dict, idf_vector)

    for word in all_words:
        idx = term_index_dict[word]
        if word not in tokenize(input_doc):
            assert doc_obj.vector[idx] == 0.0, f"{word} should have zero TF-IDF"
