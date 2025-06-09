from utils.tokenize import tokenize

corpus = [
    "Jeg lukter vondt", 
    "Dette er en frosk", 
    "Kvakk sier anda"
    
]

tokenized = [] 
for doc in corpus: 
    tokenized.append(tokenize(doc))