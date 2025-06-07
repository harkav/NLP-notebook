import math
import pytest
from utils.euclidean_distance import euclidean_distance


def test_invalid_input(): 
    # unequal length, x > y
    with pytest.raises(ValueError): 
        euclidean_distance([0.0], [1.0, 0.2])
    with pytest.raises(ValueError): 
        euclidean_distance([1.0, 0.2], [2.0])
    # one empty vector
    with pytest.raises(ValueError): 
        euclidean_distance([], [1])
    
    # test works on proper vectors
    result = euclidean_distance([0.0], [0.1])
    assert result
    
    # complains with wrong num of arguments
    with pytest.raises(TypeError): 
        euclidean_distance([0.0])
    with pytest.raises(TypeError): 
        euclidean_distance([0.0], [0.0], [0.0])
        
    
    
    
def test_euclidean_distance_calculates_properly(): 
    
    # edge case, zero vectors
    
    assert euclidean_distance([0.0], [0.0]) == 0.0
    
    # edge case, identical vectors
    
    assert euclidean_distance([1.0, 2.2], [1.0, 2.2]) == 0.0
    
    # edge case, equal sum vectors, but different places. 
    # i.e. is euclidean distance a commutative operation.
    
    result = euclidean_distance([0.2, 1.0], [1.0, 0.2])
    assert math.isclose(
        result, 
        1.13137085, 
        rel_tol=1e-5
    ), print(result)
    
    # Attempt at a standard vector
    result = euclidean_distance([0.2, 3.0, 1.4], [2.0, 2.7, 1.3])
    assert math.isclose(
        result, 
        1.827566688, 
        rel_tol=1e-5
        
    ), print(result)
    
    # all int vectors
    
    result = euclidean_distance([1, 2, 3], [3, 2, 4])
    
    assert math.isclose(
        result, 
        2.236067977,
        rel_tol=1e-5
    ), print(result)