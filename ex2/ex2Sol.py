import nltk
import numpy as np
from ex2.HMMBigramClass import HMMBigramTagger as HMM
from ex2.MostLikelyTagBaseline import MostLikelyTagBaseline as MLT

# <editor-fold desc="Q2.a">
def getTaggedSentences():
    tagged_sents = np.array(nltk.corpus.brown.tagged_sents())
    corpus_size = tagged_sents.size
    # splits the data into 2 sets - train and test set
    test_sentences = tagged_sents[round(corpus_size * 0.9):corpus_size]
    train_sentences = tagged_sents[:round(corpus_size * 0.9)]
    return train_sentences, test_sentences


# </editor-fold>

# <editor-fold desc="Q2.b">
def base(train_sentences, test_sentences):
    b = MLT()
    b.train(train_sentences=train_sentences)
    error = b.test(test_sentences=test_sentences)
    return error


# </editor-fold>

# <editor-fold desc="Q2.c">

def HMMbigramTagger(train_sentences, test_sentences):
    tagger = HMM()
    tagger.train(train_sentences=train_sentences)
    error = tagger.test(test_sentences=test_sentences)
    return error

# </editor-fold>

# <editor-fold desc="Q2.d">

def HMMbigramTaggerWithSmooth(train_sentences, test_sentences):
    tagger = HMM(delta=1)
    tagger.train(train_sentences=train_sentences)
    error = tagger.test(test_sentences=test_sentences)
    return error

# </editor-fold>

# <editor-fold desc="Q2.e">

def HMMbigramTaggerWithPseudo(train_sentences, test_sentences,pseudo_words_to_tag):
    tagger = HMM(pseudo_words_to_tag=pseudo_words_to_tag)
    tagger.train(train_sentences=train_sentences)
    error = tagger.test(test_sentences=test_sentences)
    return error

# </editor-fold>

# <editor-fold desc="Q2.e iii">

def HMMbigramTaggerWithPseudoAndSmooth(train_sentences, test_sentences,pseudo_words_to_tag):
    tagger = HMM(pseudo_words_to_tag=pseudo_words_to_tag, delta=1)
    tagger.train(train_sentences=train_sentences)
    error = tagger.test(test_sentences=test_sentences)
    return error

# </editor-fold>



