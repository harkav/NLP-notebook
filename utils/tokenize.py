import spacy



nlp = spacy.load("en_core_web_sm")


def tokenize(document: str) -> list[str]:
    """Tokenize a doc using spacy"""
    doc = nlp(document)
    processed = [
        token.text.lower() for token in doc if token.is_alpha and not token.is_stop
    ]
    return processed
