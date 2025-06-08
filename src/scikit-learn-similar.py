import sys
from pathlib import Path 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def create_tf_idfs(corpus: list[str], input_doc: str):
    vectorizer = TfidfVectorizer()
    corpus_vectors = vectorizer.fit_transform(corpus)

    input_vector = vectorizer.transform([input_doc])  # wrap in list

    similarities = cosine_similarity(input_vector, corpus_vectors)

    most_similar_index = similarities.argmax()
    most_similar_score = similarities[0, most_similar_index]
    most_similar_doc = corpus[most_similar_index]

    return most_similar_doc, most_similar_score

     


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

    most_similar_doc, most_similar_score = create_tf_idfs(corpus_list, input_doc_text)
    print(f"Most similar document:\n{most_similar_doc[:200]}")
    print(f"Similarity score: {most_similar_score:.4f}")


if __name__ == "__main__": 
    main()