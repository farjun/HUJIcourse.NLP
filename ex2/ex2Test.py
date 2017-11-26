from ex2.HMMBigramClass import *
from ex2.ex2Sol import *
words,test = getTaggedSentences()
a = HMMBigramTagger()
a.train(words)
a.tag(["the"])

