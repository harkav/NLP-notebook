

docs = ["kvakk jeg heter Harald", "dette er en hund", "dette er en øl"]

all_words = {word for doc in docs for word in doc.split()}


dic = {}

for word in all_words:
    count = 0
    for doc in docs: 
        if word in doc.split(): 
            if word not in dic.keys(): 
                count += 1
    dic[word] = count        
                
print(all_words)
print(dic)