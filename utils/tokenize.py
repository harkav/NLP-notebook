import spacy
import time as t

before_nlp_load = t.time()

# Load the SpaCy English tokenizer

nlp = spacy.load("en_core_web_sm")

after_nlp_load = t.time()

difference = after_nlp_load - before_nlp_load

print(f"NLP load: Difference is {difference}s")

def tokenize(document: str) -> list[str]:
    """Tokenize a doc using spacy"""
    before_nlp_use = t.time()
    doc = nlp(document)
    after_nlp_use = t.time()
    nlp_use_difference = after_nlp_use - before_nlp_use
    print(f"NLP use: Difference is {nlp_use_difference}s")
    before_processed = t.time()
    processed = [token.text.lower() for token in doc if token.is_alpha and not token.is_stop]
    after_processed = t.time()
    processed_difference = after_processed - before_processed
    print(f"Processed: Difference is {processed_difference}")
    return processed



