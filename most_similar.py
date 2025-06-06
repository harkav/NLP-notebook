from typing import NamedTuple
import numpy as np
from cosine_similarity import cosine_similarity
from tf_idf import tf_idf
from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# TODO think about optimalizations, plenty of room for improvements

def tokenize(doc: str) -> list[str]:
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

