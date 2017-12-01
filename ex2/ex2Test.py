import ex2.ex2Sol as ex2Sol
import numpy as np

if __name__ == '__main__':
    train , test = ex2Sol.getTaggedSentences()
    # retNormal = ex2Sol.HMMbigramTagger(train_sentences=train,test_sentences=test)
    # print("loss - normal :{Normal}".format(Normal=retNormal))
    retSmooth = ex2Sol.HMMbigramTaggerWithSmooth(train_sentences=train,test_sentences=test)
    print("loss - Smooth :{Smooth}".format(Smooth = retSmooth))

    pass
