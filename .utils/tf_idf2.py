from nltk.tokenize import word_tokenize

def set_up_tf_idf_context(): 
    pass 

def document_frequency(corpus: list[list[str]]) -> dict[str, int]:
    
    df_dict = {}
    
    
    for doc in corpus: 
        for word in doc: 
            df_dict[word] = (df_dict.get(word) or 0) +1 
    return df_dict
     
    
def term_frequency(term : str, doc : list[str]) -> int: 
    return  doc.count(term)
    




def tf_idf2(corpus : list[list[str]]): 
    
    set_of_all_words = {word for doc in corpus for word in doc}
    df_dict = document_frequency(corpus)
    
    term_index_mapping = {term : index for term, index in enumerate(set_of_all_words)}
    
    N = len(set_of_all_words)
    