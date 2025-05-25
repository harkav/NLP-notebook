import math


#TODO think about punctuation.

def term_frequency(term : str, document: str) -> int:
    """
    Find term frequency in a document. 
    
    Args: 
        term (str): the term in question.
        document (str): the document, represented as a str.
        
    Returns:
        frequency (int): the count of term in document.

    """ 
    words_in_doc = [word.lower() for word in document.split()]
    count = 0
    
    for word in words_in_doc: 
        if word == term.lower():
            count += 1
    return count
    
    
    


def document_frequency(term: str, documents: list[str]) -> int: 
    """
    Find the num of docs that contains the term. 
    
    Args: 
        term (str): the term.
        documents (list[str]): the documents of the corpus.
        
    Returns: 
        Frequency (int): the num of occurrences of term in the documents.
    
    """
    count = 0
    for doc in documents: 
        doc_split = [word.lower() for word in doc.split()]
        if term in doc_split: 
            count += 1
            
    return count


def inverse_document_frequency(term: str, documents: list[str]) -> float: 
    """
    Finds the inverse document frequency of the term.
    
    Args: 
        term (str): the term.
        documents (list[str]): the documents.
        
    Returns:
        The inverse document frequency of the term.

    
    """
    # TODO: think about zero division. 
    
    return math.log(len(documents) / document_frequency(term, documents))


def tf_idf(term, single_document, documents) -> float: 
     
    return term_frequency(term, single_document) * inverse_document_frequency(term, documents)


