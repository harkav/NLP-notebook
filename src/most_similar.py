import math
import numpy as np
from utils.cosine_similarity import cosine_similarity
from utils.tokenize import tokenize
from pathlib import Path
import sys
from dataclasses import dataclass
from collections import Counter



@dataclass(frozen=True)
class Document:
    """An immutable (kind of) wrapper class for the documents"""

    content: str
    vector: np.ndarray



def create_tf_idf_context(corpus: list[str]) -> tuple[set[str], dict[str, int], np.ndarray]:
    """
    Creates the context needed for calculating TF-IDFs. 
    
    
    Tests run, tests/create_document_frequency_dict
    
    Args: 
        corpus (list[str]): all the corpus that the model will be built on.
    
    Returns: 
        all_words (set[str]): the set of all words in corpus.
        term_index_dict (dict[str, int]): a dict that maps term to an index that should be the same in the idf-np.ndarray.
        idf_vector (np.ndarray): the idf vectors. 
    
    """
    
    # Get the set of all words in corpus
    all_words = {word for doc in corpus for word in tokenize(doc)}

    # Map the words to an index
    term_index_dict = {term: index for index, term in enumerate(sorted(all_words))}

    # Create the df
    df = create_document_frequency_dict(all_words, corpus)

    N = len(corpus)  # We need N for the inverse document frequency.

    # Here we begin calculating the inverse document frequency for all the words and store them in a np arr.
    idf_vector = np.zeros(len(all_words))

    for term, idx in term_index_dict.items():
        idf_vector[idx] = math.log(N /(1 +  df[term]))

    # Return what We've made as a tuple
    return (all_words, term_index_dict, idf_vector)




def create_object_new(doc: str, all_words: set[str], term_index_dict: dict[str, int], idf_vector: np.ndarray) -> Document:
    """
    Create Document objects from the tf-idf context.
    
    Args: 
        doc (str): a text document as a str.
        all_words (set[str]): all the words of the corpus.
        term_index_dict (dict[str, int]): a dict mapping term to an index that is shared with the np.ndarray
        idf_vector (np.ndarray): a numpy array representing the idf-vector for the corpus.
    
    Returns: 
        Document : the Document object that has been generated.
    """
    
    vector = np.zeros(len(all_words))
    tokenized = tokenize(doc)
    tf_counter = Counter(tokenized)
    for word, tf in tf_counter.items():
        term_index = term_index_dict.get(word)
        if term_index is not None:
            vector[term_index] = tf * idf_vector[term_index]
    return Document(content=doc, vector=vector)



def create_document_frequency_dict(all_words: set[str], corpus: list[str]) -> dict[str, int]:
    """
    Readable method for finding the document frequencies of a corpus.
    
    Tests run, tests/test_create_document_frequency_dict

    Args:
        all_words (set): the set of all words in the corpus.
        corpus (list[str]): the whole corpus.

    Returns:
        a dict of the document frequencies.
    """

    df = {}
    
    tokenized_corpus = []
    
    for doc in corpus: 
        tokenized_corpus.append(tokenize(doc))

    for word in all_words:
        count = 0
        for d in tokenized_corpus:
            if word in d:
                count += 1
        df[word] = count
    return df


def most_similar(input_document: str, corpus: list[str]) -> tuple[str, float, Document, Document, dict[str, int]]:
    """
    Does the comparison, create Document objects based on the corpus. 
    
    Args: 
        input_document (str): the document that will be compared against the corpus.
        corpust (list[str]): the corpus.
        
    Returns: 
        tuple[str, float, Document, Document, dict[str, str]]:
            str : text content of most similar doc
            float: similarity score of the most similar doc
            Document: the Document object created from the input document.
            Document: the Document object created from the most similar doc in corpus.
            
    """
    
    all_words, term_index_dict, idf_vector = create_tf_idf_context(corpus)

    corpus_document_objects = [
        create_object_new(doc, all_words, term_index_dict, idf_vector) for doc in corpus
    ]

    input_doc = create_object_new(input_document, all_words, term_index_dict, idf_vector)

    highest = ("", -2)
    most_similar_doc_obj = None

    for corp_doc in corpus_document_objects:
        similarity = cosine_similarity(corp_doc.vector, input_doc.vector)
        if similarity > highest[1]:
            highest = (corp_doc.content, similarity)
            most_similar_doc_obj = corp_doc

    return highest[0], highest[1], input_doc, most_similar_doc_obj, term_index_dict

    # eturn (all_words, term_index_dict, idf_vector)
    

def print_top_tfidf(doc, term_index_dict, top_n=10):
    """ Chatgpt wrote this whilst troubleshooting, not sure whether I should keep it"""
    
    index_term = {i: term for term, i in term_index_dict.items()}
    sorted_indices = np.argsort(doc.vector)[::-1]
    print("\nTop TF-IDF terms:")
    for i in sorted_indices[:top_n]:
        print(f"{index_term[i]:>15}: {doc.vector[i]:.4f}")

def validate_files(dir_path: str, input_path: str) -> None:
    """
    Validates the corpus directory and the input file.
    
    Tests run, tests/test_validate_files

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
        raise NotADirectoryError(
            f"Directory {dir_path_object} does not exist or is not a directory"
        )

    files = []
    for file in dir_path_object.iterdir():
        files.append(check_for_empty_file(file))

    # should check if there is at least one non-empty file
    if not any(files):
        raise ValueError("All the files in the directory are empty")

    if not input_path_object.is_file():
        raise FileNotFoundError(
            f"Input file {input_path} is not a file or could not be found"
        )

    if not check_for_empty_file(input_path_object):
        raise ValueError("Cannot check an empty file")

    return True


def check_for_empty_file(file_path: Path) -> bool:
    """Returns True if the size of the file is > 0, False otherwise
    
    Args:
        file_path (Path): a path to a file.
        
    Returns: 
        bool. 
    """
    return file_path.stat().st_size > 0


def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <corpus_directory> <input_file>")
        sys.exit(1)

    dir_path, input_file_path = sys.argv[1], sys.argv[2]
    validate_files(dir_path, input_file_path)

    dir_path = Path(dir_path)
    corpus_list = [p.read_text("UTF-8") for p in dir_path.iterdir()]

    with open(input_file_path, "r", encoding="UTF-8") as input_file:
        input_doc_text = input_file.read()

    most_similar_doc_content, similarity, input_doc, most_similar_doc, term_index_dict = most_similar(input_doc_text, corpus_list)

    print(f"Most similar doc: {most_similar_doc_content}")
    print(f"Similarity score: {similarity:.4f}")

    print("=== Input document top TF-IDF terms ===")
    print_top_tfidf(input_doc, term_index_dict)

    print("=== Most similar document top TF-IDF terms ===")
    print_top_tfidf(most_similar_doc, term_index_dict)

if __name__ == "__main__":
    main()
