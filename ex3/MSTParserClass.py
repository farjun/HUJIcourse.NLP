import numpy as np
import ex3.MSTAlgorithem as MSTAlgorithem


class MSTParser:
    def __init__(self):
        # conputes which tuples will have the value 1 in the sentence
        # for example - sentences_words_dic[sentence1][(word1,word2)] = 1 iff word1 word2 were in sentence1
        self.sentences_words_dic = dict()
        self.word_weight = dict()
        self.tag_weight = dict()

        # maps from a value to the feature index.
        self.vocabulary_dic = dict()
        self.tag_dic = {'ROOT': 0}

    # <editor-fold desc="Pre-Processing">
    def generateVocabulery(self, train: np.ndarray, test: np.ndarray) -> None:
        wordIndex, tagIndex = self.addWords(len(self.vocabulary_dic), len(self.tag_dic), train)
        self.addWords(wordIndex, tagIndex, test)

    def addWords(self, wordIndex, tagIndex, sentences):
        for i in range(sentences.size):
            sentence = sentences[i]
            nodes = sentence.nodes
            for j in range(1, len(nodes)):
                word = nodes[j]['word']
                tag = nodes[j]['tag']
                if word not in self.vocabulary_dic:
                    self.vocabulary_dic[word] = wordIndex
                    wordIndex += 1
                if tag not in self.tag_dic:
                    self.tag_dic[tag] = tagIndex
                    tagIndex += 1
        return wordIndex, tagIndex

    def get_weight(self, word1, word2):
        pass

    def get_full_graph_from_dict(self, sentence_dict):
        total_indexes = len(sentence_dict)
        arcs = []
        for fromIndex in range(total_indexes):
            word1 = sentence_dict[fromIndex]['word']
            for toIndex in range(total_indexes):
                if fromIndex == toIndex:
                    continue
                word2 = sentence_dict[toIndex]['word']
                arcs.append(MSTAlgorithem.Arc(fromIndex, self.get_weight(word1, word2), toIndex))

        return arcs

    # </editor-fold>



    def train(self, train_sentences):
        pass

    def generateArcs(self, sentence) -> [MSTAlgorithem.Arc]:
        totalIndexes = len(sentence)
        arcs = []
        for fromIndex in range(totalIndexes):
            for toIndex in range(totalIndexes):
                if toIndex in sentence[fromIndex]['deps']:
                    arcs.append(MSTAlgorithem.Arc(fromIndex, 1, toIndex))
                else:
                    arcs.append(MSTAlgorithem.Arc(fromIndex, 0, toIndex))

    def getWordBigram(self, sentence, fromNode, toNode):
        return 1 if toNode['address'] in fromNode['deps'] else 0

    def getPOSBigram(self, sentence, fromNode, toNode):
        tags = set([sentence[i]['tag'] for i in fromNode['deps']])
        return 1 if toNode['tag'] in tags else 0

    # <editor-fold desc="Getters & Setters">
    def get_feature(self, sentence, word1, word2):
        if sentence in self.sentences_words_dic:
            if (word1, word2) in self.sentences_words_dic[sentence]:
                return 1
        return 0

    def getWordFeatureIndex(self, word) -> int:
        return self.vocabulary_dic[word]

    def getWordsCount(self) -> int:
        return len(self.vocabulary_dic)

    def getTagsCount(self):
        return len(self.tag_dic)

    def getWordsWeight(self, w1, w2):
        if w1 not in self.word_weight:
            self.word_weight[w1] = dict()
        if w2 not in self.word_weight[w1]:
            self.word_weight[w1][w2] = 0
        return self.word_weight[w1][w2]

    def setWordsWeight(self, w1, w2, weight):
        if w1 not in self.word_weight:
            self.word_weight[w1] = dict()
        self.word_weight[w1][w2] = weight

    def getTagsWeight(self, t1, t2):
        if t1 not in self.tag_weight:
            self.tag_weight[t1] = dict()
        if t2 not in self.tag_weight[t1]:
            self.tag_weight[t1][t2] = 0
        return self.tag_weight[t1][t2]

    def setTagsWeight(self, t1, t2, weight):
        if t1 not in self.tag_weight:
            self.tag_weight[t1] = dict()
        self.tag_weight[t1][t2] = weight

    # </editor-fold>




    def dummyFunction(self):
        pass
