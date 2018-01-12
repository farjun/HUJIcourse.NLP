import spacy
import wikipedia

nlp_model = spacy.load('en')
page = wikipedia.page('Brad Pitt').content
analyzed_page = nlp_model(page)