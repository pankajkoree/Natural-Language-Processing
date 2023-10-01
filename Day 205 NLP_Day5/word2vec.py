# Word 2 Vec

from sklearn.feature_extraction.text import CountVectorizer

vectorizer=CountVectorizer()
data_corpus=["guru99 is the best sitefor online tutorials. I love to visit guru99."]
vocabulary=vectorizer.fit(data_corpus)
X= vectorizer.transform(data_corpus)
print(X.toarray())
print(vocabulary.get_feature_names_out)

### CBOW (continous bag of words)

import nltk

nltk.download('all')

import gensim
from nltk.corpus import abc

model = gensim.models.Word2Vec(abc.sents())
X = list(model.wv.index_to_key)  
data = model.wv.most_similar('science')  
print(data)


[{"tag": "welcome",
"patterns": ["Hi", "How are you", "Is any one to talk?", "Hello", "hi are you available"],
"responses": ["Hello, thanks for contacting us", "Good to see you here"," Hi there, how may I assist you?"]

        },
{"tag": "goodbye",
"patterns": ["Bye", "See you later", "Goodbye", "I will come back soon"],
"responses": ["See you later, thanks for visiting", "have a great day ahead", "Wish you Come back again soon."]
        },

{"tag": "thankful",
"patterns": ["Thanks for helping me", "Thank your guidance", "That's helpful and kind from you"],
"responses": ["Happy to help!", "Any time!", "My pleasure", "It is my duty to help you"]
        },
        {"tag": "hoursopening",
"patterns": ["What hours are you open?", "Tell your opening time?", "When are you open?", "Just your timing please"],
"responses": ["We're open every day 8am-7pm", "Our office hours are 8am-7pm every day", "We open office at 8 am and close at 7 pm"]
        },

{"tag": "payments",
"patterns": ["Can I pay using credit card?", " Can I pay using Mastercard?", " Can I pay using cash only?" ],
"responses": ["We accept VISA, Mastercard and credit card", "We accept credit card, debit cards and cash. Please don’t worry"]
        }
   ]


import json
json_file ='300 Days         [ NLP ...... ]/Day 205 NLP_Day5/intents.json'
with open('intents.json','r') as f:
    data = json.load(f)



import pandas as pd
df = pd.DataFrame(data)
df['patterns'] = df['patterns'].apply(', '.join) 

import string
from nltk.corpus import stopwords
from textblob import Word
stop = stopwords.words('english')

df['patterns'] = df['patterns'].apply(lambda x:' '.join(x.lower() for x in x.split()))

df['patterns']= df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation))
                                        
df['patterns']= df['patterns'].str.replace('[^\w\s]','')
df['patterns']= df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
df['patterns'] = df['patterns'].apply(lambda x:' '.join(x for x in x.split() if not x in stop))
df['patterns'] = df['patterns'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


from gensim.models import Word2Vec

Bigger_list = []
for i in df['patterns']:
    li = i.split()
    Bigger_list.append(li)

Model = Word2Vec(sentences=Bigger_list, vector_size=300, window=5, min_count=1, workers=4)


#list of libraries used by the code
import string
from gensim.models import Word2Vec
import logging
from nltk.corpus import stopwords
from textblob import Word
import json
import pandas as pd
#data in json format
json_file = 'intents.json'
with open('intents.json','r') as f:
    data = json.load(f)
#displaying the list of stopwords
stop = stopwords.words('english')
#dataframe
df = pd.DataFrame(data)

df['patterns'] = df['patterns'].apply(', '.join)
# print(df['patterns'])
#print(df['patterns'])
#cleaning the data using the NLP approach
print(df)
df['patterns'] = df['patterns'].apply(lambda x:' '.join(x.lower() for x in x.split()))
df['patterns']= df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if x not in string.punctuation))
df['patterns']= df['patterns'].str.replace('[^\w\s]','')
df['patterns']= df['patterns'].apply(lambda x: ' '.join(x for x in x.split() if  not x.isdigit()))
df['patterns'] = df['patterns'].apply(lambda x:' '.join(x for x in x.split() if not x in stop))
df['patterns'] = df['patterns'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
#taking the outer list
bigger_list=[]
for i in df['patterns']:
    li = list(i.split(" "))
    bigger_list.append(li)
#structure of data to be taken by the model.word2vec
print("Data format for the overall list:",bigger_list)
#custom data is fed to machine for further processing
model = Word2Vec(bigger_list, min_count=1,vector_size=300,workers=4)
#print(model)


model.save("word2vec.model")
model.save("model.bin")

model = Word2Vec.load('model.bin')

similar_words = model.wv.most_similar('thanks')
print(similar_words)

dissimlar_words = model.wv.doesnt_match('See you later, thanks for visiting'.split())
print(dissimlar_words)

similarity_two_words = model.wv.similarity('please','see')
print("Please provide the similarity between these two words:")
print(similarity_two_words)

similar = model.wv.similar_by_word('kind')
print(similar)