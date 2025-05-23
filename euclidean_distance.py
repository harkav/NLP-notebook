import math 
"""
# Euclidean Distance

The Euclidean distance between two vectors **a** and **b** is calculated as:

$$
d(a, b) = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}
$$

This function computes the distance for vectors of equal length.
"""


def euclidean_distance(vector1: list[float], vector2 : list[float]) -> float :
    """
    A method for finding the euclidean distance of two vectors.
    
    Args: 
        vector1 (list[float]): a list representing a vector.
        vector2 (list[float]): a list representing a different vector. 
        
    Returns: 
        sum (float): the square root of the sum of the calculation.
        
    Raises: 
        ValueError: If the vectors are not of the same length. 
    """
    
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must be of the same length")
        
    
    total = 0 
    for v1, v2 in zip(vector1, vector2): 
        total += (v1 - v2)**2
    return math.sqrt(total)

