import pytest
from euclidean_distance import euclidean_distance


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
    