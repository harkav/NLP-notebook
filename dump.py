import math

def tokenize(doc : str ) -> list[str]: 
    """Simple tokenizer, no stop-word removal"""
    return [word.lower() for word in doc.split()] 


def term_frequency(term: str, document: str) -> int:
    """
    Count how many times "term" appears in "document". 
    """
    words_in_doc = tokenize(document)
    return words_in_doc.count(term.lower()) 

def document_frequency(term: str, documents: list[str]) -> int:
    """
    Count how many documents contain "term".
    """
    count = 0
    for doc in documents:
        tokenized = tokenize(doc)
        if term.lower() in tokenized:
            count += 1
    return count

def inverse_document_frequency(term: str, documents: list[str]) -> float:
    """
    Compute IDF for "term" across the "documents".
    """
    N = len(documents)
    df = document_frequency(term, documents)
    return math.log(N / (1 + df))  # Smoothing with +1 to avoid division by zero

def tf_idf(term: str, document: str, documents: list[str]) -> float:
    """
    Compute TF-IDF score of "term" in "document", relative to "documents" as corpus.
    """
    tf = term_frequency(term, document) / len(tokenize(document))
    idf = inverse_document_frequency(term, documents)
    return tf * idf


corpus = ["the cat sat on the mat", "the duck is not happy", "the dog and the cat are friends"] 



all_words = sorted({word for doc in corpus for word in tokenize(doc)})


print(all_words).


doc = corpus[0]

for word in sorted(doc.split()): 
    print(word, tf_idf(word, doc, corpus))

