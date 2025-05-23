import pytest
from dotproduct import dot_product


def test_validate_input():
    with pytest.raises(ValueError):
        dot_product([1, 2], [1])
    with pytest.raises(ValueError):
        dot_product([1, 2], [1, 2, 3])
    

def test_yields_zero_from_empty_vectors(): 
    assert dot_product([], []) == 0   
    
        
def test_works_with_both_floats_and_ints(): 
    assert dot_product([1], [1.2]) == 1.2
    assert dot_product([1.7], [2]) == 3.4        
    

def test_dotproduct(): 
    assert dot_product([1, 2, 3], [2, 2, 2]) == 12 
    assert dot_product([1, 1, 7], [2, 3, 6]) == 47 
    assert dot_product([0], [1]) == 0
    assert dot_product([1.0, 2.0, 3.0], [2, 2, 2]) == 12.0
    assert dot_product([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == 12.0
    assert dot_product([1, 2, 3], [2.0, 2.0, 2.0]) == 12
     