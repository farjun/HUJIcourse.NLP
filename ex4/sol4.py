import spacy
import wikipedia
from ex4.TreeExtractor import TreeExtractor

nlp_model = spacy.load('en')
page = wikipedia.page('Brad Pitt').content

analyzed_page = nlp_model(page)


a = TreeExtractor()
a.extract_propn_tags(analyzed_page=analyzed_page)

