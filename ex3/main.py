import nltk
import numpy as np
from nltk.corpus import dependency_treebank


def getTaggedSentences() -> [np.ndarray, np.ndarray]:
    # to_conll = mystery function that outputs the items like this:template = '{word}\t{tag}\t{head}\n'
    # todo check if its better to work with the trees
    # tagged_sents = np.array([np.array(sentence.to_conll(3), dtype=np.str_) for sentence in dependency_treebank.parsed_sents()])
    tagged_sents = np.array([dependency_treebank.index(i) for i in range(3914)])
    corpus_size = tagged_sents.size

    # splits the data into 2 sets - train and test set
    train_sentences, test_sentences = np.split(tagged_sents, [round(corpus_size * 0.9)])
    return train_sentences, test_sentences

print(dependency_treebank.parsed_sents()[0].nodes.items())
