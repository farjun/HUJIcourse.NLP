import nltk
import numpy as np
from nltk.corpus import dependency_treebank

import ex3.MSTParserClass as MSTParserClass


def getTaggedSentences() -> [np.ndarray, np.ndarray]:
    tagged_sents = np.array(dependency_treebank.parsed_sents())
    corpus_size = tagged_sents.size
    # splits the data into 2 sets - train and test set
    train_sentences, test_sentences = np.split(tagged_sents, [round(corpus_size * 0.9)])
    return train_sentences, test_sentences


if __name__ == '__main__':
    from time import time
    s = time()
    train_sentences, test_sentences = getTaggedSentences()
    parser = MSTParserClass.MSTParser()
    parser.train(train_sentences)
    print(parser.test(test_sentences))
    print(time()-s)