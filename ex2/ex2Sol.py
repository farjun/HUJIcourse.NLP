import nltk
import numpy as np
from ex2.HMMBigramClass import HMMBigramTaggerCLass

# <editor-fold desc="Q2.a">
def getTaggedSents():
    tagged_sents = np.array(nltk.corpus.brown.tagged_sents())
    corpus_size = tagged_sents.size
    # splits the data into 2 sets - train and test set
    test_sents = tagged_sents[round(corpus_size * 0.9):corpus_size]
    train_sents = tagged_sents[:round(corpus_size * 0.9)]
    return train_sents, test_sents

# </editor-fold>

# <editor-fold desc="Q2.b">
class MostLikelyTagBaseline:
    def __init__(self):
        self._words_tag_count = dict()
        pass

    def train(self, train_sentences):
        for sentence in train_sentences:
            for word, tag in sentence:
                if word not in self._words_tag_count:
                    self._words_tag_count[word] = dict()
                if tag not in self._words_tag_count[word]:
                    self._words_tag_count[word][tag] = 1
                else:
                    self._words_tag_count[word][tag] += 1

    def _getBestTag(self, word):
        if word not in self._words_tag_count:
            return "NN"
        tagsMap = self._words_tag_count[word]
        bestTag = "NN"
        bestCount = 0
        for tag, count in tagsMap.items():
            if count > bestCount:
                bestTag = tag
                bestCount = count
        return bestTag

    def test(self, test_sentences):
        wrong = 0
        right = 0
        for sentence in test_sentences:
            for word, actualTag in sentence:
                expectedTag = self._getBestTag(word)
                if expectedTag == actualTag:
                    right += 1
                else:
                    wrong += 1
        return (wrong/(right+wrong))

def base():
    train_sentences, test_sentences = getTaggedSents()
    b = MostLikelyTagBaseline()
    b.train(train_sentences=train_sentences)
    error = b.test(test_sentences=test_sentences)
    return error

# </editor-fold>

# <editor-fold desc="Q2.c">
def HMMbigramTagger(train_sents, test_sents):
    tagger = HMMBigramTaggerCLass()
    tagger.train(train_sents)
    tagger.tag("signal")

# </editor-fold>

