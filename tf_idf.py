import math
from nltk.tokenize import word_tokenize



def term_frequency(term : str, document: str) -> int:
    """
    Find term frequency in a document. 
    
    Args: 
        term (str): the term in question.
        document (str): the document, represented as a str.
        
    Returns:
        frequency (int): the count of term in document.

    """ 
    assert isinstance(document, str)
    words_in_doc = tokenize(document)
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
    assert isinstance(documents, list)
    assert len(documents) > 0
    assert isinstance(documents[0], str)
    count = 0
    for doc in documents: 
        tokenized = word_tokenize(doc)
        tokenized = [word.lower() for word in tokenized]
        if term.lower() in tokenized: 
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
    assert isinstance(term, str)
    assert isinstance(documents, list)
    assert len(documents) > 0
    assert isinstance(documents[0], str)
    
    
    N = len(documents)
    df = document_frequency(term, documents)
    
    
    return math.log(N / (1 + df))


def tf_idf(term: str, single_document: str, documents: list[str]) -> float: 
     
    #
    """
    Returns term frequency * inverse document frequency.
    
    Args: 
        term (str): the term you're trying to find the tf-idf for.
        single_document (str): the document you're trying to find the tf-idf for.
        documents (list[str]): the whole corpus.
        
    Returns: 
        tf-idf (float): the tf-idf based on the input.
        
        
    """
    
    assert isinstance(term, str)
    assert isinstance(single_document, str)
    assert isinstance(documents, list)
    assert len(documents) > 0    
    assert isinstance(documents[0], str)
   
    
    return term_frequency(term, single_document) * inverse_document_frequency(term, documents)


def tokenize(doc: str) -> list[str]:
    return [word.lower() for word in word_tokenize(doc) if word.isalnum()]

