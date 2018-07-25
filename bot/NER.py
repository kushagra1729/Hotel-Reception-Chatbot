import spacy
nlp = spacy.load('en')
doc=nlp("a cheap hotel in the London")
for ent in doc.ents:
	print(ent.text,ent.label_)
