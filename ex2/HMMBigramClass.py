import numpy as np
class HMMBigramTaggerCLass:
    def __init__(self):
        self.trainset = {}
        self.counter_dic = {}
        self.possibleTags = {}

    def setDics(self, train_sentences):
        """
        sets the dics to 0 and adds a key value to them
        :param train_sentences:
        :return:
        """
        for sentence in train_sentences:
            for i in range(len(sentence) - 1):
                # all pairs of words with their tags
                self.counter_dic[(sentence[i][0], sentence[i][1])] = 0
                # all pairs of tags
                self.counter_dic[(sentence[i][1], sentence[i + 1][1])] = 0
                # all tags
                self.counter_dic[sentence[i][1]] = 0
                self.possibleTags[sentence[i][1]] = 1

    def countWords(self, train_sentences):
        """

        :param train_sentences:
        :return:
        """
        for sentence in train_sentences:
            for i in range(len(sentence) - 1):
                # all pairs of words with their tags
                self.counter_dic[(sentence[i][0], sentence[i][1])] += 1
                # all pairs of tags
                self.counter_dic[(sentence[i][1], sentence[i + 1][1])] += 1
                # all tags
                self.counter_dic[sentence[i][1]] += 1

    def train(self, train_sentences):
        self.setDics(train_sentences)
        self.countWords(train_sentences)


    def setPorbMatrix(self,number_of_possible_tags,sentence):
        n = len(sentence)
        return np.zeros((number_of_possible_tags,n))


    def tag(self,sentence):
        n = len(sentence)
        Sk = []
        list_of_possible_tags = list(self.possibleTags.keys())

        Sk.append(["*"])
        for i in range(1,n):
            Sk.append(list_of_possible_tags.copy())
        Sk.append(["*"])

        probMatrix = self.setPorbMatrix(n,sentence)


