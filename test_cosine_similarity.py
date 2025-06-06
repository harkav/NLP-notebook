import pytest 
import math
from cosine_similarity import cosine_similarity


def test_validate_input(): 
    with pytest.raises(ValueError): 
        cosine_similarity([0.0], [1.0, 2])
    with pytest.raises(ValueError): 
        cosine_similarity([0.0, 2.2, 3.0], [1.0, 2])
    

def test_cosine_similarity_calculates_properly(): 
    #Edge case, to vectors of [0.0]
    
    result = cosine_similarity([0.0], [0.0])
    assert result == 0.0, print(result)
    
    #Edge case, identical vectors. 
    
    result = cosine_similarity([1.0, 1.0], [1.0, 1.0])
    assert round(result, 5) == 1.0, print(result)    
    

