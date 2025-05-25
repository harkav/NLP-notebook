from tf_idf import pre_process
import pathlib
"""
Project for creating vectors out of a set of document

"""


def find_all_unique_words_in_folder(dir_path: str) -> set[str]: 
    """
    """
    
    dirpath = pathlib.Path(dir_path)
    
    all_words = set()
    
    for maybe_file in dirpath.iterdir():
        if pathlib.Path.is_file(maybe_file): 
            file_as_str = get_content_as_string(maybe_file)
            tokens = pre_process(file_as_str)
            tokens_lower = (token.lower() for token in tokens) # is this a proper set comprehension? 
            for e in tokens_lower: 
                all_words.add(e)
    return all_words
            
            
            
def get_content_as_string(filename : pathlib.Path) -> str: 

    
    with open(filename, "r") as f: 
        file_as_str = f.read()
        #print(file_as_str) works here. 
        return file_as_str
         
        
print(find_all_unique_words_in_folder("./test-docs"))


"""
ChatGpt suggested workflow for a plagiarism finder: 

Input: List of documents.

Preprocess: Tokenize and normalize each document.

Build vocabulary: Gather all unique words across corpus.

Vectorize: For each document, compute TF-IDF vector based on the vocabulary.

Compare: Compute cosine similarity between every pair of document vectors.

Detect: Flag pairs with similarity over a chosen threshold.

Output: Present flagged pairs and similarity scores.

"""