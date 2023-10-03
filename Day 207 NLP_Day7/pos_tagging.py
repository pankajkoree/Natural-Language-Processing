# Part of Speech Tagging

import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp(u"I will google about facebook")

print(doc.text)

print(doc[-1])

print(doc[2].pos_)

print(doc[2].tag_)

print(spacy.explain('VB'))

for word in doc:
    print(word.text,"------>", word.pos_,word.tag_,spacy.explain(word.tag_))


doc2 = nlp(u"I left the room")
for word in doc2:
    print(word.text,"------>", word.pos_,word.tag_,spacy.explain(word.tag_))



doc3 = nlp(u"to the left of the room")
for word in doc3:
    print(word.text,"------>", word.pos_,word.tag_,spacy.explain(word.tag_))

doc4 = nlp(u"I read books on history")
for word in doc4:
    print(word.text,"------>", word.pos_,word.tag_,spacy.explain(word.tag_))


doc5 = nlp(u"I have read a book on history")
for word in doc5:
    print(word.text,"------>", word.pos_,word.tag_,spacy.explain(word.tag_))


doc6 = nlp(u"The quick brown fox jumped over the lazy dog")

from spacy import displacy

displacy.render(doc6,style='dep',jupyter=True)

options={
    'distance':80,
    'compact':True,
    'color':'#fff',
    'bg':'#00a65a'
}

displacy.render(doc6,style='dep',jupyter=True,options=options)