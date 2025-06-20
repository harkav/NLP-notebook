from utils.tf_idf import (
    document_frequency,
    create_tf_idf_context,
    tokenize,
    term_frequency,
    
)


#TODO didn't test tf_idf_2 :/ Don't think it's necessary to do so directly, but could be an idea to do so. 

global_corpus = [
    ["the cat sat on the mat"],
    ["the duck is not happy"],
    ["the cat and dog are friends"],
]


def test_document_frequency():
    all_words = set()

    for doc in global_corpus:
        all_words.update(tokenize(doc[0]))

    doc_freq_dict = document_frequency(all_words, global_corpus)
    assert doc_freq_dict["the"] == 3
    assert doc_freq_dict["cat"] == 2
    assert doc_freq_dict["mat"] == 1
    assert doc_freq_dict["and"] == 1


def test_idf_vector():
    term_index_mapping, idf = create_tf_idf_context(global_corpus)
    assert idf[term_index_mapping["cat"]] == 0
    assert abs(idf[term_index_mapping["sat"]] - 0.40546) < 1e-5
    assert abs(idf[term_index_mapping["the"]] - (-0.28768)) < 1e-5


def test_tf():
    test_doc = ["Beer, beer, beer. I want beer"]
    tokenized = tokenize(test_doc[0])
    assert term_frequency("beer", tokenized) == 4
    assert term_frequency("i", tokenized) == 1
    assert term_frequency("want", tokenized) == 1


def test_tf_idf():
    term_index_mapping, idf = create_tf_idf_context(global_corpus)
    doc1_tokenized = tokenize(global_corpus[0][0])
    assert term_frequency("cat", doc1_tokenized) * idf[term_index_mapping["cat"]] == 0
    assert (
        abs(
            term_frequency("mat", doc1_tokenized) * idf[term_index_mapping["mat"]]
            - 0.40546
        )
        < 1e-5
    )
    assert (
        abs(term_frequency("the", doc1_tokenized)) * idf[term_index_mapping["the"]]
        - (-0.57536)
        < 1e-5
    )
    
    
