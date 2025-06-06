import pytest 
import math 
from tf_idf import term_frequency, tf_idf, document_frequency, inverse_document_frequency, tokenize



test_docs =[
    "kvakk, kvakk", 
    "jeg heter Harald, kvakk", 
    "Dette er en setning", 
    "New York er en by"
]


def test_validate_input_term_frequency(): 
    with pytest.raises(AssertionError): 
        term_frequency("Kvakk", ["kvekk"])
        

def test_tokenize(): 
    assert tokenize("kvakk") == ["kvakk"]
    assert tokenize("New York") == ["new", "york"]
    assert tokenize("Jeg liker øl") == ["jeg", "liker", "øl"]

def test_term_frequency_counts_correctly(): 
    assert term_frequency("kvakk", test_docs[0]) == 2
    assert term_frequency("kvakk", "kvekk") == 0
    assert term_frequency("kvakk", "kvakk, kvakk, kvakk, kvekk") == 3
    

def test_validate_input_document_frequency(): 
    with pytest.raises(AssertionError): 
        document_frequency("kvakk", "kvekk")
    with pytest.raises(AssertionError): 
        document_frequency("kvakk", [])
    with pytest.raises(AssertionError): 
        document_frequency("kvakk", [1, 2])
        

def test_document_frequency_counts_correctly(): 
    assert document_frequency("kvakk", test_docs) == 2
    assert document_frequency("dette", test_docs) == 1
    assert document_frequency("Erna Solberg", test_docs) == 0
    assert document_frequency("New York", test_docs) == 0
    
    

def test_validate_input_inverse_document_frequency():
    with pytest.raises(AssertionError): 
        inverse_document_frequency(1, test_docs)
    with pytest.raises(AssertionError): 
        inverse_document_frequency("kvakk", "kvekk")
    with pytest.raises(AssertionError): 
        inverse_document_frequency("kvakk",[])
    with pytest.raises(AssertionError): 
        inverse_document_frequency("kvakk", [0, 1])
        
def test_inverse_frequency_calculates_correctly(): 
    assert math.isclose(
        inverse_document_frequency("kvakk", test_docs),
        0.287679572, 
        rel_tol=1e-5
        
        
    )
    assert math.isclose(
        inverse_document_frequency("Erna", test_docs), 
        1.386294361, 
        rel_tol= 1e-5
    )
    assert math.isclose(
        inverse_document_frequency("kvakk", ["kvakk", "kvakk"]), 
        -0.405465108,
        rel_tol=1e-5
        
        
    )


def test_tf_idf_validate_input(): 
    with pytest.raises(AssertionError): 
        tf_idf(1, test_docs[0], test_docs)
    with pytest.raises(AssertionError): 
        tf_idf("kvakk", ["kvekk"], test_docs)
    with pytest.raises(AssertionError): 
        tf_idf("kvakk", "kvekk", [])
    with pytest.raises(AssertionError): 
        tf_idf("kvakk", "kvakk", [0, 1])
        

def test_tf_idf_calculates_correctly(): 
    assert math.isclose(
        tf_idf("Kvakk", test_docs[0], test_docs), 
        0.575359144,
        rel_tol=1e-5
        )
    assert math.isclose(
        tf_idf("Erna", test_docs[0], test_docs), 
        0, 
        rel_tol=1e-5
        
    )
    assert math.isclose(
        tf_idf("Kvakk", "Kvakk", ["kvakk", "kvakk"]), 
        -0.405465108 * 1,
        rel_tol=1e-5
        
    )
    
