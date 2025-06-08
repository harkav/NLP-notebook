import math
from nltk.tokenize import word_tokenize
import numpy as np

"""
Attempt of implementing tf-idf in a way that doesn't make redundant calcs
"""


def document_frequency(corpus: list[list[str]]) -> dict[str, int]:
    df_dict = {}
    for doc in corpus:
        for word in set(doc):  # Use set to avoid duplicates
            df_dict[word] = df_dict.get(word, 0) + 1
    return df_dict



def term_frequency(term: str, doc: list[str]) -> int:
    return doc.count(term / (0.0001 + len(doc))) # zero division? Shouldn't be possible, but just in case.


def create_tf_idf_context(corpus: list[list[str]]):
    set_of_all_words = {word for doc in corpus for word in doc}
    df_dict = document_frequency(corpus)

    term_index_mapping = {term: idx for idx, term in enumerate(set_of_all_words)}
    idf_vector = np.zeros(len(set_of_all_words))
    N = len(corpus)

    for term, idx in term_index_mapping.items():
        idf_vector[idx] = math.log(N / (1 + df_dict[term]))

    return (term_index_mapping, idf_vector)

def tf_idf_2(set_of_all_words : set[str], term_index_mapping : dict[str, int]
             , document  :list[str], idf_vector : np)-> np.ndarray: 
    
    return_vector = np.zeros(len(set_of_all_words))
    
    for word in document: 
        if word in term_index_mapping.keys(): 
            index_word = term_index_mapping[word]
            value = term_frequency(word, document) * idf_vector[index_word]
            return_vector[index_word] = value 
    return return_vector 
    