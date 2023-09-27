# Natural Language Processing

import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

example_string = """
Muad'Dib learned rapidly because his first training was in how to learn.
And the first lesson of all was the basic trust that he could learn.
It's shocking to find how many people do not believe they can learn,
and how many more believe learning to be difficult."""

sent_tokenize(example_string)

word_tokenize(example_string)

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

worf_quote = "Sir, I protest. I am not a merry man!"


words_in_quote = word_tokenize(worf_quote)
words_in_quote

stop_words = set(stopwords.words("english"))

filtered_list = []

for word in words_in_quote:
   if word.casefold() not in stop_words:
        filtered_list.append(word)



filtered_list = [
    word for word in words_in_quote if word.casefold() not in stop_words
]

filtered_list

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

string_for_stemming = """
The crew of the USS Discovery discovered many discoveries.
Discovering is what explorers do."""

words = word_tokenize(string_for_stemming)


words

stemmed_words = [stemmer.stem(word) for word in words]

stemmed_words

from nltk.tokenize import word_tokenize

sagan_quote = """
If you wish to make an apple pie from scratch,
you must first invent the universe."""

words_in_sagan_quote = word_tokenize(sagan_quote)

import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Your list of words
words_in_sagan_quote = ["The", "Cosmos", "is", "within", "us."]

# Perform part-of-speech tagging
nltk.pos_tag(words_in_sagan_quote)


import nltk

# Download the tagsets data
nltk.download('tagsets')

# Use nltk.help.upenn_tagset() to access the Penn Treebank POS tagset information
nltk.help.upenn_tagset()


jabberwocky_excerpt = """
'Twas brillig, and the slithy toves did gyre and gimble in the wabe:
all mimsy were the borogoves, and the mome raths outgrabe."""

words_in_excerpt = word_tokenize(jabberwocky_excerpt)

nltk.pos_tag(words_in_excerpt)

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

import nltk

# Download the WordNet data
nltk.download('wordnet')

# Create a WordNet lemmatizer
lemmatizer = nltk.WordNetLemmatizer()

# Lemmatize the word "scarves"
lemmatized_word = lemmatizer.lemmatize("scarves")
print(lemmatized_word)


string_for_lemmatizing = "The friends of DeSoto love scarves."

words = word_tokenize(string_for_lemmatizing)

words

lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

lemmatized_word

lemmatizer.lemmatize("worst")

from nltk.tokenize import word_tokenize

lotr_quote = "It's a dangerous business, Frodo, going out your door."

words_in_lotr_quote = word_tokenize(lotr_quote)
words_in_lotr_quote

nltk.download("averaged_perceptron_tagger")
lotr_pos_tags = nltk.pos_tag(words_in_lotr_quote)
lotr_pos_tags


grammar = "NP: {<DT>?<JJ>*<NN>}"

chunk_parser = nltk.RegexpParser(grammar)

tree = chunk_parser.parse(lotr_pos_tags)

tree.draw()

lotr_pos_tags

grammar = """
Chunk: {<.*>+}
       }<JJ>{"""

chunk_parser = nltk.RegexpParser(grammar)

tree = chunk_parser.parse(lotr_pos_tags)

tree.draw()

nltk.download("maxent_ne_chunker")
nltk.download("words")
tree = nltk.ne_chunk(lotr_pos_tags)


tree.draw()

tree = nltk.ne_chunk(lotr_pos_tags, binary=True)
tree.draw()


quote = """
Men like Schiaparelli watched the red planet—it is odd, by-the-bye, that
for countless centuries Mars has been the star of war—but failed to
interpret the fluctuating appearances of the markings they mapped so well.
All that time the Martians must have been getting ready.

During the opposition of 1894 a great light was seen on the illuminated
part of the disk, first at the Lick Observatory, then by Perrotin of Nice,
and then by other observers. English readers heard of it first in the
issue of Nature dated August 2."""

import nltk
from nltk.tokenize import word_tokenize

def extract_ne(quote, language='english'):
    words = word_tokenize(quote, language=language)
    tags = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tags, binary=True)
    return set(
        " ".join(i[0] for i in t)
        for t in tree
        if hasattr(t, "label") and t.label() == "NE"
    )


extract_ne(quote)

nltk.download("book")
from nltk.book import *

text8.concordance("man")

text8.concordance("woman")

text8.dispersion_plot(
    ["woman", "lady", "girl", "gal", "man", "gentleman", "boy", "guy"]
)


text2.dispersion_plot(["Allenham", "Whitwell", "Cleveland", "Combe"])

from nltk import FreqDist

frequency_distribution = FreqDist(text8)
print(frequency_distribution)

frequency_distribution.most_common(20)

meaningful_words = [
    word for word in text8 if word.casefold() not in stop_words
]

frequency_distribution = FreqDist(meaningful_words)

frequency_distribution.most_common(20)

frequency_distribution.plot(20, cumulative=True)

text8.collocations()

lemmatized_words = [lemmatizer.lemmatize(word) for word in text8]

new_text = nltk.Text(lemmatized_words)

new_text.collocations()

