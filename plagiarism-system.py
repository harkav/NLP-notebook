from cosine_similarity import cosine_similarity
from itertools import product

class Plagiarism_System: 
    
    def __init__(self, input_file : str, document_dir : str): 
        self._input_file = input_file
        self._document_dir = document_dir
        self._vocab = [] # will contain all the unique terms
        self._word_counts = dict()
        # mapping.... Or mappings.
        self._doc_chunks_input = [] # might be a good idea to keep these separate. 
        self._doc_chunks = []
        self.preprocess(self, self._input_file, self._document_dir)
        
        
        
    def preprocess(self, input_file, document_dir): 
        pass 
    # preprocess the docs, fill out the constructor. 
    
    
    def compare(self): 
        # run cos sim
        THRESHOLD = 0.8
        
        for input_chunk, db_chunk in product(self._doc_chunks_input, self._doc_chunks): 
            # chatgpt told me about itertools.product
    
            result = cosine_similarity(input_chunk.get_vector(), db_chunk.get_vector())
            if result > THRESHOLD: 
                print(f"Similarity between input doc-id {input_chunk.get_id()} and {db_chunk.get_id()} is {result}")    
    