from typing import NamedTuple
import numpy as np
from utils.cosine_similarity import cosine_similarity
from utils.tf_idf import tf_idf
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# TODO think about optimalizations, plenty of room for improvements, are you doing something several times, rather than once? 
# TODO think about the logic for files and validation. 
# TODO write some doc strings, perhaps rewrite a lot, there was a lot of hate-coding whilst making this. 



def tokenize(doc: str) -> list[str]:
    """ Wrapper method for nltk.tokenize
    
        Args: 
            doc (str) : a document represented as a str.
            
        Returns: 
            list[str]: the tokenized doc as a list of str.
    """
    return [word.lower() for word in word_tokenize(doc) if word.isalnum()]


class Document(NamedTuple):
    content: str = ""
    vector: np.array = []


def most_similar(input_document: str, corpus: list[str]) -> tuple[str, float]:
    # create wordlist

    tokenized_corpus = [tokenize(doc) for doc in corpus]
    print(tokenized_corpus)
    tokenized_as_str = corpus 
    input_tokens = tokenize(input_document)
    all_docs = tokenized_corpus
    all_words = {word for doc in all_docs for word in doc}

    mapping_dict = {i: term for i, term in enumerate(all_words)}
    mapping_reversed = {v: k for k, v in mapping_dict.items()}
    documents = []

    for doc in tokenized_corpus:
        documents.append(
            create_document_object(corpus, all_words, mapping_reversed, doc)
        )

    input_document_object = create_input_doc(
        tokenized_as_str, input_tokens, all_words, mapping_reversed
    )

    highest = -2
    return_str = ""

    for doc_object in documents:
        
        similarity = cosine_similarity(
            input_document_object.vector.tolist(), doc_object.vector.tolist()
        )
        print(similarity)
        if similarity > highest:
            highest = similarity
            return_str = doc_object.content

    return (return_str, highest)


def create_input_doc(tokenized_corpus_as_str, input_tokens, all_words, mapping_reversed):
    input_tokens_as_str = " ".join(input_tokens)
    numpy_arr = np.zeros(len(all_words))
    for token in input_tokens:
        if token in mapping_reversed:
            value = tf_idf(token, input_tokens_as_str, tokenized_corpus_as_str)
            numpy_arr[mapping_reversed.get(token)] = value
    if np.count_nonzero(numpy_arr) == 0:
        print("Warning: zero vector for doc:", input_tokens[:50])  # preview
    return Document(content=input_tokens, vector=numpy_arr)


def create_document_object(corpus, all_words, mapping_reversed, doc):
    
    numpy_arr = np.zeros(len(all_words))
    for token in doc:
        value = tf_idf(token, " ".join(doc), corpus)
        numpy_arr[mapping_reversed.get(token)] = value
        if np.count_nonzero(numpy_arr) == 0:
            print("Warning: zero vector for doc:", doc[:50])  # preview

    return Document(content=doc, vector=numpy_arr)

def validate_files(dir_path : str, input_path : str) -> None: 
    """
    Validates the corpus directory and the input file.
    
    Args: 
        dir_path (str): the path to the corpus directory as a str.
        input_path (str): the path to the input file.
        
    Raises: 
        NotADirectoryError if the dir_path is not a directory.
        ValueError either if all the corpus files are empty or the input file is empty.
        FileNotFoundError if the input_path is not a path to a file.
    """    
   
    dir_path_object = Path(dir_path)
    input_path_object = Path(input_path)

    if not dir_path_object.is_dir(): 
        raise NotADirectoryError(f"Directory {dir_path_object} does not exist or is not a directory")
    
    files = [] 
    for file in dir_path_object.iterdir(): 
        files.append(check_for_empty_file(file))
     
    # should check if there is at least one non-empty file    
    if not any(files): 
        raise ValueError("All the files in the directory are empty")
    
    if not input_path_object.is_file(): 
        raise FileNotFoundError(f"Input file {input_path} is not a file or could not be found")
    
    if not check_for_empty_file(input_path_object): 
        raise ValueError("Cannot check an empty file")
    
    return True         
            

def check_for_empty_file(file_path: Path):
    """ Returns True if the size of the file is > 0, False otherwise"""
    return file_path.stat().st_size > 0

  

def main():
    path = Path("./test-docs/")
    corpus_list = []

    for p in path.iterdir():
        with open(p) as f:
            corpus_list.append(f.read())

    with open("./input.txt", "r", encoding="UTF-8") as input_file:
        input_doc = input_file.read()
        #print(input_doc)

    print(most_similar(input_doc, corpus_list))


if __name__ == "__main__":
    main()

