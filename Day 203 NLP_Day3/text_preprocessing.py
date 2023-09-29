# Text Preprocessing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"D:\copy of htdocs\practice\Python\300 Days         [ NLP ...... ]\Day 203 NLP_Day3\archive\IMDB Dataset.csv")

print(df.shape)

# to lowercase

df['review'][3].lower()

df['review']=df['review'].str.lower()

print(df)

# removing html tags


df['review'][1]

import re
def rem_htmltags(text):
    pattern =  re.compile('<.*?>')
    return pattern.sub(r'',text)


text = 'a wonderful little production. <br /><br />the filming technique is very unassuming- very old-time-bbc fashion and gives a comforting, and sometimes discomforting, sense of realism to the entire piece. <br /><br />the actors are extremely well chosen- michael sheen not only "has got all the polari" but he has all the voices down pat too! you can truly see the seamless editing guided by the references to williams\' diary entries, not only is it well worth the watching but it is a terrificly written and performed piece. a masterful production about one of the great master\'s of comedy and his life. <br /><br />the realism really comes home with the little things: the fantasy of the guard which, rather than use the traditional \'dream\' techniques remains solid then disappears. it plays on our knowledge and our senses, particularly with the scenes concerning orton and halliwell and the sets (particularly of their flat with halliwell\'s murals decorating every surface) are terribly well done.'


rem_htmltags(text)

df['review']=df['review'].apply(rem_htmltags)

print(df)

# removing URLs

text1='https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download'
text2 = 'https://www.youtube.com/shorts/IPCFmA_04U8'

def url_rem(text):
    pattern = re.compile(r'https?://\S+|www.\.\S+')
    return pattern.sub(r'',text)

url_rem(text1)          

# Removing punctuation


import string

print(string.punctuation)

exclude = string.punctuation

def rem_punct(text):
    for char in exclude:
        text = text.replace(char,"")
    return text


text = 'String. With. Punctuation'

rem_punct(text)

# above technique will be slow if we apply it on large dataset so here's another one

def rem_punct1(text):
    return text.translate(str.maketrans('','',exclude))

rem_punct1(text1)

tw_df = pd.read_csv(r"D:\copy of htdocs\practice\Python\300 Days         [ NLP ...... ]\Day 203 NLP_Day3\archive1\labeled_data.csv")


tw_df.sample(5)

def rem_punct1(text):
    return text.translate(str.maketrans('','',exclude))


tw_df['tweet']=tw_df['tweet'].apply(rem_punct1)

import io

# chat word treatment

f = open(r"D:\copy of htdocs\practice\Python\300 Days         [ NLP ...... ]\Day 203 NLP_Day3\slang.txt")

chat_words = {}
for line in f:
    parts = line.split()
    if len(parts) == 2:
        key, value = parts
        chat_words[key] = value




print(chat_words)

def chat_conversion(text):
    new_text = []
    for w in text.split():
        if w.upper() in chat_words:
            new_text.append(chat_words[w.upper()])
        else:
            new_text.append(w)
    return " ".join(new_text)

chat_conversion('IMHO he is the best')

chat_conversion('FYI delhi is the capital of india')


# Spelling correction

from textblob import TextBlob

incorrect_text = "wat the hull are yu guys doinr?"
TextBlb = TextBlob(incorrect_text)

TextBlb.correct().string

# removing stop words

from nltk.corpus import stopwords

stopwords.words('spanish')

def remove_stopwords(text):
    new_text = []
    
    for word in text.split():
        if word in stopwords.words('english'):
            new_text.append('')
        else:
            new_text.append(word)
    x = new_text[:]
    new_text.clear()
    return " ".join(x)


remove_stopwords('probably my all-time favorite movie, a story of selflessness, sacrifice and dedication to a noble cause, but it\'s not preachy or boring. it just never gets old, despite my having seen it some 15 or more times')


df.head()

df['review'][0:100].apply(remove_stopwords)

# removing emojis

import re
def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

remove_emoji("Loved the movie. It was 😘😘")

remove_emoji("Lmao 😂😂")

import emoji
print(emoji.demojize('Python is 🔥'))

print(emoji.demojize('Loved the movie. It was 😘'))

### Tokenization

# using split function

# word tokenization
sent1 = 'I am going to delhi'
sent1.split()

# sentence tokenization
sent2 = 'I am going to delhi. I will stay there for 3 days. Let\'s hope the trip to be great'
sent2.split('.')

# Problems with split function
sent3 = 'I am going to delhi!'
sent3.split()

sent4 = 'Where do think I should go? I have 3 day holiday'
sent4.split('.')

# using regular expression

import re
sent3 = 'I am going to delhi!'
tokens = re.findall("[\w']+", sent3)
tokens


text = """Lorem Ipsum is simply dummy text of the printing and typesetting industry? 
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
when an unknown printer took a galley of type and scrambled it to make a type specimen book."""
sentences = re.compile('[.!?] ').split(text)
sentences


# using NLTK

from nltk.tokenize import word_tokenize,sent_tokenize

sent1 = 'I am going to visit delhi!'
word_tokenize(sent1)

text = """Lorem Ipsum is simply dummy text of the printing and typesetting industry? 
Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
when an unknown printer took a galley of type and scrambled it to make a type specimen book."""

sent_tokenize(text)

sent5 = 'I have a Ph.D in A.I'
sent6 = "We're here to help! mail us at nks@gmail.com"
sent7 = 'A 5km ride cost $10.50'

word_tokenize(sent5)

word_tokenize(sent6)

word_tokenize(sent7)

# using spacy

import spacy
nlp = spacy.load('en_core_web_sm')

doc1 = nlp(sent5)
doc2 = nlp(sent6)
doc3 = nlp(sent7)
doc4 = nlp(sent1)

for token in doc4:
    print(token)


# stemming

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
def stem_words(text):
    return " ".join([ps.stem(word) for word in text.split()])

sample = "walk walks walking walked"
stem_words(sample)

text = 'probably my alltime favorite movie a story of selflessness sacrifice and dedication to a noble cause but its not preachy or boring it just never gets old despite my having seen it some 15 or more times in the last 25 years paul lukas performance brings tears to my eyes and bette davis in one of her very few truly sympathetic roles is a delight the kids are as grandma says more like dressedup midgets than children but that only makes them more fun to watch and the mothers slow awakening to whats happening in the world and under her own roof is believable and startling if i had a dozen thumbs theyd all be up for this movie'
print(text)

stem_words(text)

import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

sentence = "He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun."
punctuations="?:!.,;"
sentence_words = nltk.word_tokenize(sentence)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)

sentence_words
print("{0:20}{1:20}".format("Word","Lemma"))
for word in sentence_words:
    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word,pos='v')))


