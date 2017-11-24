class HMMBigramTaggerCLass:
    def __init__(self):
        self.trainset = {}
        self.probability_dic = {}
        self.mona_counter_dic = {}
        self.mehane_counter_dic = {}

#
    def setDics(self, train_sentences):
        for sentence in train_sentences:
            for i in range(len(sentence)-1):
                self.mona_counter_dic[(sentence[i][0],sentence[i+1][0])] = 0
                self.mona_counter_dic[(sentence[i][0], "*")] = 0


    def train(self, train_sentences):
        self.setDics(train_sentences)

