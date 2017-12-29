import numpy as np
import ex3.MSTAlgorithem as MSTAlgorithem
import nltk

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

        # error vars
        self.total_edges_checked = 0
        self.total_edges_right = 0



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



    # </editor-fold>

    def test(self,test_sentences):
        for i in range(test_sentences.size):
            cur_sentence_tree = test_sentences[i]
            full_graph = self.get_full_graph_from_dict(cur_sentence_tree.nodes)
            mst_graph = MSTAlgorithem.min_spanning_arborescence(full_graph, 0)
            self.calculate_error(mst_graph, cur_sentence_tree.nodes)
        return self.total_edges_right / self.total_edges_checked

    def calculate_error(self, our_tree, real_tree):
        our_tree_dict = MSTAlgorithem.turn_output_to_dict(our_tree)
        for i in range(len(real_tree)):
            cur_word_dict = real_tree[i]
            for deps in cur_word_dict['deps']['']:
                if (i,deps) in  our_tree_dict:
                    self.total_edges_right += 1
                self.total_edges_checked += 1

    def train(self, train_sentences):
        for i in range(train_sentences.size):
            cur_sentence_tree = train_sentences[i]
            full_graph = self.get_full_graph_from_dict(cur_sentence_tree.nodes)
            mst_graph = MSTAlgorithem.min_spanning_arborescence(full_graph,0)
            self.set_new_weights_by_trees(mst_graph, cur_sentence_tree.nodes)

    def get_full_graph_from_dict(self, sentence_dict):
        total_indexes = len(sentence_dict)
        arcs = []
        for fromIndex in range(total_indexes):
            word1 = sentence_dict[fromIndex]
            for toIndex in range(total_indexes):
                if fromIndex == toIndex:
                    continue
                word2 = sentence_dict[toIndex]
                arcs.append(MSTAlgorithem.Arc(fromIndex, -1*self.getWordsWeight(word1['word'], word2['word']), toIndex))
        return arcs

    def set_new_weights_by_trees(self, our_tree, real_tree):
        # todo check if the numbers here are compatible with our numbers
        # todo check what to do with the root!
        # add arcs of real tree by iterating the nodes (hard)
        for i in range(len(real_tree)):
            cur_word_dict = real_tree[i]
            # adds all the edges of cur_word_dict by iterating on cur_word_dict['deps'][''] (which is for
            # some magical reason the nodes neighbors)
            for deps in cur_word_dict['deps']['']:
                # adds 1 to the weight 'vector' of cur_word_dict['word'] = current word , real_tree[deps]['word'] = cur neightbor
                self.setWordsWeight(cur_word_dict['word'],real_tree[deps]['word'],1)
                self.setTagsWeight(cur_word_dict['tag'], real_tree[deps]['tag'], 1)

        # add arcs of our tree by iterating the arcs (easy)
        for i in range(1,len(our_tree)):
            from_word_index = our_tree[i].tail
            to_word_index = our_tree[i].head
            self.setWordsWeight(real_tree[from_word_index]['word'],real_tree[to_word_index]['word'],-1)
            self.setTagsWeight(real_tree[from_word_index]['tag'], real_tree[to_word_index]['tag'], -1)




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
        if w1 not in self.word_weight or w2 not in self.word_weight[w1]:
            return 0
        return self.word_weight[w1][w2]

    def setWordsWeight(self, w1, w2, weight):
        if w1 not in self.word_weight:
            self.word_weight[w1] = dict()
        if w2 not in self.word_weight[w1]:
            self.word_weight[w1][w2] = weight
        else:
            self.word_weight[w1][w2] += weight

    def getTagsWeight(self, t1, t2):
        if t1 not in self.tag_weight or t2 not in self.tag_weight[t1]:
            return 0
        return self.tag_weight[t1][t2]

    def setTagsWeight(self, t1, t2, weight):
        if t1 not in self.tag_weight:
            self.tag_weight[t1] = dict()
        if t2 not in self.tag_weight[t1]:
            self.tag_weight[t1][t2] = weight
        else:
            self.tag_weight[t1][t2] += weight

    # </editor-fold>
