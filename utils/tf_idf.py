import math
import numpy as np
import spacy



nlp = spacy.load("en_core_web_sm")


def tokenize(document: str) -> list[str]:
    """Tokenize a doc using spacy, leave stop words"""
    doc = nlp(document)
    processed = [
        token.text.lower() for token in doc if token.is_alpha]
    return processed

def document_frequency(all_words : set[str], corpus: list[list[str]]) -> dict[str, int]:
    """ Calculate document frequency for all terms in corpus, return doc freq dict"""
    
    df_dict = {}
    # tokenize docs
    tokenized_all_docs = [] 
    for doc in corpus: 
        #extract doc as str
        doc = doc[0]
        tokens = [word for word in tokenize(doc)]
        tokenized_all_docs.append(tokens)
        
    # count df 

    for term in all_words:
        counter = 0
        for doc in tokenized_all_docs: 
            if term in doc: 
                counter += 1
        df_dict[term] = counter  
                    
    return df_dict

def term_frequency(term: str, doc: list[str]) -> int:
    """Return count of term in doc """
    return doc.count(term)

def create_tf_idf_context(corpus: list[list[str]]) -> tuple[dict[str, int], list[float]]:
    """ Creates the term_index_mapping dict and the idf_vector"""
    
    all_words = {word 
                 for doc in corpus 
                 for sentence in doc
                for word in tokenize(sentence)
                }
    df_dict = document_frequency(all_words, corpus)
    term_index_mapping = {term: idx for idx, term in enumerate(sorted(all_words))}  # ✅ Sort terms
    idf_vector = np.zeros(len(all_words))
    N = len(corpus)
    for term, idx in term_index_mapping.items():
        idf_vector[idx] = math.log(N / (1 + df_dict.get(term, 0)))
    return term_index_mapping, idf_vector

def tf_idf(term_index_mapping: dict[str, int], document: list[str], idf_vector: np.ndarray) -> np.ndarray:
    """ Creates tf-idf vector for document"""
    
    # first we create the empty vector which will be the tf-idf vector of a document.
    tfidf_vector = np.zeros(len(idf_vector))
    # iterate over the terms in the token
    for word in document:
        # make sure that we map the term to the right index of the idf_vector
        if word in term_index_mapping:
            idx = term_index_mapping[word]
            tf = term_frequency(word, document)
            # calculate the tf_idf score, update vector
            tfidf_vector[idx] = tf * idf_vector[idx]
    return tfidf_vector
