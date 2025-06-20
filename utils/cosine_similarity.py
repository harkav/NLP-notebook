

from .length_normalization import length_normalization
from .dotproduct import dot_product



def cosine_similarity(vector1: list[float], vector2 : list[float]) -> float :
    """
    A method for finding the cosine similarity of two vectors.
    
    Args: 
        vector1 (list[float]): a list representing a vector.
        vector2 (list[float]): a list representing a different vector. 
        
    Returns: 
        dotproduct (float): the dot product of the normalized vectors.
        
    Raises: 
        ValueError: If the vectors are not of the same length. 
    """
    
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
    if all(v == 0 for v in vector1) or all(v == 0 for v in vector2): 
        return 0.0
        
    
    vector1_normalized = length_normalization(vector1)
    vector2_normalized = length_normalization(vector2)
    result = dot_product(vector1_normalized, vector2_normalized)
    
    return round(result, ndigits= 7) 