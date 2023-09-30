# text representation

import numpy as np
import pandas as pd


df = pd.DataFrame({'text':['people watch campusx','campusx watch campusx','people write comment','campusx write comment'],'output':[1,1,0,0]})

print(df)

from sklearn.feature_extraction.text import CountVectorizer

# bag of words or unigrams

cv = CountVectorizer()

bow = cv.fit_transform(df['text'])

# vocab
cv.vocabulary_

print(bow[0].toarray())
print(bow[1].toarray())

cv.transform(['campusx watch and write comment of campusx']).toarray()

# bigrams

cv = CountVectorizer(ngram_range=(2,2))

bow = cv.fit_transform(df['text'])

# vocab
cv.vocabulary_


# unigrams + bigrams

cv = CountVectorizer(ngram_range=(1,2))
bow = cv.fit_transform(df['text'])
# vocab
cv.vocabulary_


# trigrams

cv = CountVectorizer(ngram_range=(3,3)) #cant be more than 3 words as a sentence contains max of 3 words

bow = cv.fit_transform(df['text'])
# vocab
cv.vocabulary_

# combo of uni, bi and tri grams

cv = CountVectorizer(ngram_range=(1,3)) #cant be more than 3 words as a sentence contains max of 3 words

bow = cv.fit_transform(df['text'])
# vocab
cv.vocabulary_

# TF IDF

print(df)

from sklearn.feature_extraction.text import TfidfVectorizer

tdidf = TfidfVectorizer()
print("TFIDF = ",tdidf.fit_transform(df['text']).toarray())

# --------------------------------

print("\n\n IDF = ",tdidf.idf_)

# IDF = log[n/df(t)]+1 (if smooth_idf=false)
print("\n\n Get features = ",tdidf.get_feature_names_out)

