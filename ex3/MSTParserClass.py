import numpy as np


class MSTParser:
    def __init__(self):
        # conputes which tuples will have the value 1 in the sentence
        # for example - sentences_words_dic[sentence1][(word1,word2)] = 1 iff word1 word2 were in sentence1
        self.vocabulery_dic = dict()
        self.sentences_words_dic = dict()
        self.tag_dic = dict()
        self.weights = np.zeros(1)


    def generateVocabulery(self,train:np.ndarray,test:np.ndarray):
        index = self.addWordsToVocabulery(0, train)
        self.addWordsToVocabulery(index, test)

    def addWordsToVocabulery(self, index, sentences):
        for i in range(sentences.size):
            sentence = sentences[i]
            nodes = sentence.nodes
            for j in range(1, len(nodes)):
                word = nodes[j]['word']
                if word not in self.vocabulery_dic:
                    self.vocabulery_dic[word] = index
                    index += 1
        return index

    def train(self, train_sentences):
        for sentence in train_sentences:
            self.sentences_words_dic[sentence] = dict()

            for i in range(len(sentence) - 1):
                self.sentences_words_dic[sentence][(sentence[i], sentence[i + 1])] = 1
                # todo put tags here too
                # todo think of how to init the weights vector better
                # for now we save the vocabulary
                self.vocabulery_dic[sentence[i]] = 1
            self.vocabulery_dic[sentence[len(sentence)]] = 1


    def get_feature(self, sentence, word1, word2):
        if sentence in self.sentences_words_dic:
            if (word1, word2) in self.sentences_words_dic[sentence]:
                return 1
        return 0

    def get_weight(self,sentence,word1,word2):
        pass
