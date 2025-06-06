import pytest 

from tf_idf import term_frequency, tf_idf, document_frequency


test_docs =[
    "kvakk, kvakk", 
    "jeg heter Harald, kvakk", 
    "Dette er en setning", 
    "New York er en by"
]


def test_validate_input_term_frequency(): 
    with pytest.raises(AssertionError): 
        term_frequency("Kvakk", ["kvekk"])
        

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
    
    
def test_tf_idf_validate_input(): 
    with pytest.raises(AssertionError): 
        tf_idf(1, test_docs[0], test_docs)
    with pytest.raises(AssertionError): 
        tf_idf("kvakk", ["kvekk"], test_docs)
    with pytest.raises(AssertionError): 
        tf_idf("kvakk", "kvekk", [])
    with pytest.raises(AssertionError): 
        tf_idf("kvakk", "kvakk", [0, 1])
        
