import ex2.ex2Sol as ex2Sol
import numpy as np

if __name__ == '__main__':
    # train , test = ex2Sol.getTaggedSentences()
    # ret = ex2Sol.HMMbigramTagger(train_sentences=train,test_sentences=test)
    a = ["s" for i in range(10)]
    b = ["s" for i in range(5)] + ["ss" for i in range(5)]
    print(
        np.array(b).astype(np.str)
        ==
        np.array(a).astype(np.str)
    )
    pass
