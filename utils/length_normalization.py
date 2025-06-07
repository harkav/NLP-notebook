import math

def length_normalization(vector: list[float]) -> list[float] :
    """
    A method for normalizing a vector.
    Tested to a precision of 1 * 10⁻².
    
    Args: 
        vector1 (list[float]): a list representing a vector.
         
        
    Returns: 
        normalized_vector (list[float]): the sum of the calculation.
        
    Raises: 
        ValueError: If it is an empty vector (like []) 
    """
    
 
    if not vector: 
        raise ValueError("Cannot normalize an empty vector ([])")
    
    total = 0
    for element in vector: 
        total += element ** 2
    length = math.sqrt(total)
    
    if length == 0: 
        return vector
    normalized_vector = [x/ length for x in vector]

    return normalized_vector


