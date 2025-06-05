from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
import re
from nltk.corpus import stopwords
import matplotlib.pyplot as plt


nltk.download("punkt")
nltk.download("stopwords")

stopwords = set(stopwords.words("norwegian"))
#print(stopwords)



def main():
    with open("epiktet-frihet.txt", "r", encoding="utf-8") as f: 
        sample = f.read()
        sample = preprocess(sample)



    counter_dict = Counter(word_tokenize(sample))
    sorted_counter_dict = counter_dict.most_common(25)

    for k in sorted_counter_dict: 
        print(k)

    
def preprocess(text : str) -> str: 
    """
    Removes html-tags from text.
    
    Args: 
        text (str): the input text.
        
    Returns: 
        cleaned_text (str): the cleaned text.
    """
    text = re.sub("<.*?>", "", text)
    text = re.sub("[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stopwords]
    return " ".join(words)


def visualize(): 
    pass 

if __name__ == "__main__": 
    main()