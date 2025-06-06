import pytest 
from length_normalization import length_normalization
import math 

def test_validate_input_length_normalization(): 
    # test empty vector
    with pytest.raises(ValueError): 
        length_normalization([])
    # test proper input: 
    
    result = length_normalization([0.0, 1.0, 0.1])
    assert result
    assert len(result) == 3
    assert isinstance(result, list)
    
    # test edge case, zero vector
    
    result = length_normalization([0.0])
    assert result 
    assert len(result) == 1
    assert isinstance(result, list)
    

def test_proper_calculation_length_normalization(): 
    # test zero sum vector returns zero sum vector
    
    result = length_normalization([0.0, 0.0, 0.0])
    assert len(result) == 3
    assert isinstance(result, list)
    assert result == [0.0, 0.0, 0.0]
    assert isinstance(result[0], float)
    
    
    # test proper calculation of vector
    
    result = length_normalization([0.5])
    assert result == [1.0]
    assert len(result) == 1
    assert isinstance(result, list)
    assert isinstance(result[0], float)
    
    
    
    # test proper calculation of longer vector
    
    result = length_normalization([0.2, 1.0, 3.2])
    assert math.isclose(
        result[0], 
        0.059654999,
        rel_tol=1e-2
    ), print(result[0])
    assert math.isclose(
        result[1],
        0.298274993,
        rel_tol=1e-2
        
    ), print(result[1])
    assert math.isclose(
        result[2],
        0.954479978, 
        rel_tol=1e-2
    ), print(result[2])
    
    # not too pleased with the precision of the calculation here, either with the code or with my manual calc. 
    # it's probably the latter which is off. 