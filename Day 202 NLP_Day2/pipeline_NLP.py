# PIPELINE in NLP (Natural Language Processing)

# Step 2 :- Text Preparation

# Removing HTML tags

sample_text = """<!DOCTYPE html><html><body><p>This text is normal.</p><p><b>This text is bold.</b></p></body></html>"""


print(sample_text)

import re
def striphtml(data):
    p = re.compile('<.*?>')
    return p.sub('', data)


striphtml(sample_text)


# Unicode Normalization

emoji_text = """Happy birthday mitra🥹🫶🏻 janma din ko muri muri suvakamana🥳 dherai samjhana ra maya🩵 ramrari padhdai janu jailey ramro marks leunu😂 ali barsa ko struggle ho ajhai keep working hard🖤 ramailo garnu aaja ko din moj gara"""


print(emoji_text)

emoji_text.encode('utf-8')


# Spell checker
incorrect_text =  """His manner was not effusive. It seldom was; but he was glad, I

think, to see me. With hardley a word spoken, but with a kindly

eye, he waved me to an armchair, threw across his case of cigars,

and indicated a spirit case and a gasogene in the corner. Then he

stood before the fire and looked me over in his singular

introspctive fashion.

“Wedlock suits you,” he remarked. “I think, Watson, that you have

put on seven and a half pounds since I saw you.”

“Seven!” I answered.

“Indeed, I should have thought a little more. Just a triffle more,

I fancy, Watson. And in practice again, I observe. You did not

tell me that your intended to go into harness.”

“Then, how do you now?”

“I see it, I deduce it. How do I know that you have been getting

yourself very wet lately, and that you have a most clumsy and

careless servent girl?”

“My dear Holmes,” said I, “this is to much. You would certainly

have been burned, had you lived a few centuries ago. It is true

that I had a country walk on Thursday and came home in a dreadful

mess, but as I have changed my cloths I can’t imagine how you

deduce it. As to Mary Jane, she is incorrigable, and my wife has

given her notice, but their, again, I fail to see how you work it

out.”

He chuckled to himself and rubbed his long, nervous hands

together."""


from textblob import TextBlob

TextBlb = TextBlob(incorrect_text)

TextBlb.correct()


# Step -3 Text Preprocessing

# Tokenization
dummy = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum"

print(dummy)

import nltk
nltk.download('punkt')

from nltk.tokenize import sent_tokenize,word_tokenize

sents = sent_tokenize(dummy)

print(sents)

for sent in sents:
    print(word_tokenize(sent))

