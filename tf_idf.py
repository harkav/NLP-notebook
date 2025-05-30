import math
import re 


def pre_process(document : str) -> list[str]:
    """
    Separates words from puncts.
    
    Args: 
        document (str): the document.
        
    Returns: 
        document_as_list (list[str]): the document as a list 
    
    """ 
    
    return re.findall(r"\b\w+\b", document)



def term_frequency(term : str, document: str) -> int:
    """
    Find term frequency in a document. 
    
    Args: 
        term (str): the term in question.
        document (str): the document, represented as a str.
        
    Returns:
        frequency (int): the count of term in document.

    """ 
    words_in_doc = [word.lower() for word in pre_process(document)]
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
        doc_split = [word.lower() for word in pre_process(doc)]
        if term.lower() in doc_split: 
            count += 1
            
    return count

#TODO maybe consider doing with the doc frequency during the creation of the chuncks. 


def inverse_document_frequency(term: str, documents: list[str]) -> float: 
    """
    Finds the inverse document frequency of the term.
    
    Args: 
        term (str): the term.
        documents (list[str]): the documents.
        
    Returns:
        The inverse document frequency of the term.

    
    """
    
    N = len(documents)
    df = document_frequency(term, documents)
    
    
    return math.log(N / (1 + df))


def tf_idf(term: str, single_document: list[str], documents: list[str]) -> float: 
     
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
    return term_frequency(term, single_document) * inverse_document_frequency(term, documents)


