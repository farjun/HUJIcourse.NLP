import nltk
import numpy as np
from ex2.HMMBigramClass import HMMBigramTagger as HMM
from ex2.MostLikelyTagBaseline import MostLikelyTagBaseline as MLT


# <editor-fold desc="Q2.a">
def getTaggedSentences() -> [np.ndarray, np.ndarray]:
    tagged_sents = np.array([np.array(sentence, dtype=np.str) for sentence in nltk.corpus.brown.tagged_sents()])
    corpus_size = tagged_sents.size
    # splits the data into 2 sets - train and test set
    train_sentences, test_sentences = np.split(tagged_sents, [round(corpus_size * 0.9)])
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

def HMMbigramTaggerWithPseudo(train_sentences, test_sentences, pseudo_words):
    for j in range(len(train_sentences)):
        for i in range(len(train_sentences[j])):
            if train_sentences[j][i][0] in pseudo_words:
                train_sentences[j][i][0] = pseudo_words[train_sentences[j][i][0]]

    for j in range(len(test_sentences)):
        for i in range(len(test_sentences[j])):
            if test_sentences[j][i][0] in pseudo_words:
                test_sentences[j][i][0] = pseudo_words[test_sentences[j][i][0]]
    return HMMbigramTagger(train_sentences, test_sentences)


# </editor-fold>

# <editor-fold desc="Q2.e iii">

def HMMbigramTaggerWithPseudoAndSmooth(train_sentences, test_sentences, pseudo_words):
    for j in range(len(train_sentences)):
        for i in range(len(train_sentences[j])):
            if train_sentences[j][i][0] in pseudo_words:
                train_sentences[j][i][0] = pseudo_words[train_sentences[j][i][0]]

    for j in range(len(test_sentences)):
        for i in range(len(test_sentences[j])):
            if test_sentences[j][i][0] in pseudo_words:
                test_sentences[j][i][0] = pseudo_words[test_sentences[j][i][0]]
    return HMMbigramTaggerWithSmooth(train_sentences, test_sentences)


# </editor-fold>


def optimizedTest(train_sentences, test_sentences, getMatrix=False):
    tagger = HMM()
    tagger.train(train_sentences=train_sentences)
    errorNormal = tagger.test(test_sentences=test_sentences)

    if getMatrix:
        tagger.setFlags(delta=1, compute_confusion_matrix=1)
    else:
        tagger.setDelta(1)
    errorSmooth = tagger.test(test_sentences=test_sentences)
    if getMatrix:
        return errorNormal, errorSmooth, tagger.getConfusionMatrix(),tagger.getTags()
    else:
        return errorNormal, errorSmooth
