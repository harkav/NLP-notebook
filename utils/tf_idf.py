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
    df_dict = {}
    print(all_words)

    
    # tokenize docs
    tokenized_all_docs = [] 
    for doc in corpus: 
        #extract doc as str
        doc = doc[0]
        
        print(doc)
        print(type(doc))
        tokens = [word for word in tokenize(doc)]
        tokenized_all_docs.append(tokens)
        
    
    # count df 

    for term in all_words:
        counter = 0
        for doc in tokenized_all_docs: 
            if term in doc: 
                counter += 1
        df_dict[term] = counter  
        
    print(df_dict)
            
            
        
    return df_dict

def term_frequency(term: str, doc: list[str]) -> int:
    return doc.count(term)

def create_tf_idf_context(corpus: list[list[str]]) -> tuple[dict[str, int], list[float]]:
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

def tf_idf_2(term_index_mapping: dict[str, int], document: list[str], idf_vector: np.ndarray) -> np.ndarray:
    tfidf_vector = np.zeros(len(idf_vector))
    for word in document:
        if word in term_index_mapping:
            idx = term_index_mapping[word]
            tf = term_frequency(word, document)
            tfidf_vector[idx] = tf * idf_vector[idx]
    return tfidf_vector

# ----------- Example Usage -----------


corpus = [["The cat sat on the mat"], 
        ["The duck is not happy"], 
        ["The cat and dog are friends"]]


term_index_mapping, idf_vector = create_tf_idf_context(corpus)

for term, index in term_index_mapping.items(): 
    print(term, idf_vector[index]) 


