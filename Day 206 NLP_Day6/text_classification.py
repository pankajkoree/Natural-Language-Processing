# Text Classification

import numpy as np
import pandas as pd

df=pd.read_csv(r"D:\copy of htdocs\practice\Python\300 Days         [ NLP ...... ]\Day 206 NLP_Day6\archive\IMDB Dataset.csv")

df = df.iloc[:10000]

df.head()

df['review'][1]

df['sentiment'].value_counts()

df.isnull().sum()

df.duplicated().sum()

df.drop_duplicates(inplace=True)

df.duplicated().sum()

# Basic Preprocessing
# Remove tags
# lowercase
# remove stopwords
import re
def remove_tags(raw_text):
    cleaned_text = re.sub(re.compile('<.*?>'), '', raw_text)
    return cleaned_text

df['review'] = df['review'].apply(remove_tags)

df

df['review'] = df['review'].apply(lambda x:x.lower())

from nltk.corpus import stopwords

sw_list = stopwords.words('english')

df['review'] = df['review'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))

df

X = df.iloc[:,0:1]
y = df['sentiment']

X
y


from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

y = encoder.fit_transform(y)

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)

X_train.shape

# Applying BoW
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()

X_train_bow.shape

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train_bow,y_train)

y_pred = gnb.predict(X_test_bow)

from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,y_pred)

confusion_matrix(y_test,y_pred)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(X_train_bow,y_train)
y_pred = rf.predict(X_test_bow)
accuracy_score(y_test,y_pred)

cv = CountVectorizer(max_features=3000)

X_train_bow = cv.fit_transform(X_train['review']).toarray()
X_test_bow = cv.transform(X_test['review']).toarray()

rf = RandomForestClassifier()

rf.fit(X_train_bow,y_train)
y_pred = rf.predict(X_test_bow)
accuracy_score(y_test,y_pred)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train['review']).toarray()
X_test_tfidf = tfidf.transform(X_test['review'])

rf = RandomForestClassifier()

rf.fit(X_train_tfidf,y_train)
y_pred = rf.predict(X_test_tfidf)

accuracy_score(y_test,y_pred)

import gensim

from gensim.models import Word2Vec,KeyedVectors

path = r'D:\copy of htdocs\practice\Python\300 Days [ NLP ]\Day 206 NLP_Day6\google_new_datas\GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(path, binary=True, limit=10000)  # Adjust 'limit' as needed

model['cricket'].shape

from nltk.corpus import stopwords

sw_list = stopwords.words('english')

sw_list

# Remove stopwords

X_train = X_train['review'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))
# Remove stopwords

X_test = X_test['review'].apply(lambda x: [item for item in x.split() if item not in sw_list]).apply(lambda x:" ".join(x))

import spacy
import en_core_web_sm
# Load the spacy model. This takes a few seconds.
nlp = en_core_web_sm.load()
# Process a sentence using the model
doc = nlp(X_train.values[0])
print(doc.vector)

input_arr = []
for item in X_train.values:
    doc = nlp(item)
    input_arr.append(doc.vector)


input_arr = np.array(input_arr)

input_arr.shape

input_test_arr = []
for item in X_test.values:
    doc = nlp(item)
    input_test_arr.append(doc.vector)


input_test_arr = np.array(input_test_arr)

input_test_arr = np.array(input_test_arr)
input_test_arr.shape

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(input_arr,y_train)
y_pred = gnb.predict(input_test_arr)
accuracy_score(y_test,y_pred)
