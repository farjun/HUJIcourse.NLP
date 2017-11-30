import ex2.ex2Sol as ex2Sol
import numpy as np

if __name__ == '__main__':
    train , test = ex2Sol.getTaggedSentences()
    ret = ex2Sol.HMMbigramTagger(train_sentences=train,test_sentences=test)
    print(ret)
    pass
