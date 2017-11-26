from ex2.HMMBigramClass import *
from ex2.ex2Sol import *
words,test = getTaggedSents()
a = HMMBigramTagger()
a.train(words)

