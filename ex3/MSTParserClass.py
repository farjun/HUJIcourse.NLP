import numpy as np
import ex3.MSTAlgorithem as MSTAlgorithem
import nltk


class MSTParser:
    def __init__(self, distance_flag=False):
        # conputes which tuples will have the value 1 in the sentence
        # for example - sentences_words_dic[sentence1][(word1,word2)] = 1 iff word1 word2 were in sentence1
        # self.sentences_words_dic = dict()
        # self.word_weight = dict()
        # self.tag_weight = dict()
        self.word_distance_weight = dict()

        # total weight dict
        # self.total_word_weight = dict()
        # self.total_tag_weight = dict()

        # maps from a value to the feature index.
        # self.vocabulary_dic = dict()
        # self.tag_dic = {'ROOT': 0}

        # error vars
        self.total_edges_checked = 0
        self.total_edges_right = 0

        # flags
        self.distance_flag = distance_flag

        # testing
        self.former_feature_weight_vector = np.array([], dtype=np.float64)
        self.feature_weight_vector = np.array([], dtype=np.float64)
        self.map_words_to_vector = {}
        self.cur_feature_count = 0
        self.map_tags_to_vector = {}
        self.cur_tags_count = 0

    # <editor-fold desc="Pre-Processing">
    # def generateVocabulery(self, train: np.ndarray, test: np.ndarray) -> None:
    #     wordIndex, tagIndex = self.addWords(len(self.vocabulary_dic), len(self.tag_dic), train)
    #     self.addWords(wordIndex, tagIndex, test)

    # def addWords(self, wordIndex, tagIndex, sentences):
    #     for i in range(sentences.size):
    #         sentence = sentences[i]
    #         nodes = sentence.nodes
    #         for j in range(1, len(nodes)):
    #             word = nodes[j]['word']
    #             tag = nodes[j]['tag']
    #             if word not in self.vocabulary_dic:
    #                 self.vocabulary_dic[word] = wordIndex
    #                 wordIndex += 1
    #             if tag not in self.tag_dic:
    #                 self.tag_dic[tag] = tagIndex
    #                 tagIndex += 1
    #     return wordIndex, tagIndex

    # </editor-fold>

    # <editor-fold desc="Train">


    def train(self, train_sentences):
        for i in range(train_sentences.size):
            cur_sentence_tree = train_sentences[i]
            full_graph = self.get_full_graph_from_dict(cur_sentence_tree.nodes)
            mst_graph = MSTAlgorithem.min_spanning_arborescence(full_graph, 0)
            self.set_new_weights_by_trees(mst_graph, cur_sentence_tree.nodes)
            if i % 100 == 0:
                print("number of iterations so far : ", i, "/", train_sentences.size)

        for i in range(train_sentences.size - 1, -1, -1):
            cur_sentence_tree = train_sentences[i]
            full_graph = self.get_full_graph_from_dict(cur_sentence_tree.nodes)
            mst_graph = MSTAlgorithem.min_spanning_arborescence(full_graph, 0)
            self.set_new_weights_by_trees(mst_graph, cur_sentence_tree.nodes)
            if i % 100 == 0:
                print("number of iterations so far : ", i, "/", train_sentences.size)

        # after the training we normelize the data
        self.normelize_total_weight_dict(train_sentences.size)
        # self.word_weight = self.total_word_weight
        # self.tag_weight = self.total_tag_weight

    def get_full_graph_from_dict(self, sentence_dict):
        total_indexes = len(sentence_dict)
        arcs = []
        for fromIndex in range(total_indexes):
            word1 = sentence_dict[fromIndex]
            for toIndex in range(total_indexes):
                if fromIndex == toIndex:
                    continue
                word2 = sentence_dict[toIndex]
                arcs.append(
                    MSTAlgorithem.Arc(fromIndex, -1 *
                                      (self.getWordsWeightOPT(word1['word'], word2['word'])
                                       + self.getTagsWeightOPT(word1['tag'], word2['tag'])),
                                      toIndex))
        return arcs

    def set_new_weights_by_trees(self, our_tree, real_tree):
        # todo check what to do with the root!
        # add arcs of real tree by iterating the nodes (hard)

        a = np.zeros(self.feature_weight_vector.shape,dtype=np.float64)

        for i in range(len(real_tree)):
            cur_word_dict = real_tree[i]
            # adds all the edges of cur_word_dict by iterating on cur_word_dict['deps']['']
            for deps in cur_word_dict['deps']['']:
                # adds 1 to the weight 'vector' of cur_word_dict['word'] = current word , real_tree[deps]['word'] = cur neighbor
                a = self.setWordsWeightOPT(cur_word_dict['word'], real_tree[deps]['word'], 1,a)
                a = self.setTagsWeightOPT(cur_word_dict['tag'], real_tree[deps]['tag'], 1,a)

        # add arcs of our tree by iterating the arcs (easy)
        for i in range(1, len(our_tree) + 1):
            from_word_index = our_tree[i].tail
            to_word_index = our_tree[i].head
            a =self.setWordsWeightOPT(real_tree[from_word_index]['word'], real_tree[to_word_index]['word'], -1,a)
            a =self.setTagsWeightOPT(real_tree[from_word_index]['tag'], real_tree[to_word_index]['tag'], -1,a)


        diff = a.size - self.feature_weight_vector.size
        if diff:
            self.feature_weight_vector = np.hstack((self.feature_weight_vector,np.zeros((a.size-self.feature_weight_vector.size,))))
        self.feature_weight_vector += a

        # self.update_total_weight_dict()

    # def update_total_weight_dict(self):
    #     diff_size = self.feature_weight_vector.size - self.former_feature_weight_vector.size
    #     if diff_size:
    #         self.former_feature_weight_vector = np.hstack(
    #             (self.former_feature_weight_vector, np.array([0] * diff_size)))
    #
    #     self.feature_weight_vector = self.former_feature_weight_vector + self.feature_weight_vector
    #     self.former_feature_weight_vector = self.feature_weight_vector.copy()

        # for word1 in self.word_weight.keys():
        #     for word2 in self.word_weight[word1].keys():
        #         self.total_word_weight[word1][word2] += self.word_weight[word1][word2]
        #
        # for tag1 in self.tag_weight.keys():
        #     for tag2 in self.tag_weight[tag1].keys():
        #         self.total_tag_weight[tag1][tag2] += self.tag_weight[tag1][tag2]

    def normelize_total_weight_dict(self, N):
        self.feature_weight_vector /= (2 * N)
        # for word1 in self.word_weight.keys():
        #     for word2 in self.word_weight[word1].keys():
        #         self.total_word_weight[word1][word2] /= 2 * N
        #
        # for tag1 in self.tag_weight.keys():
        #     for tag2 in self.tag_weight[tag1].keys():
        #         self.total_tag_weight[tag1][tag2] /= 2 * N

    # </editor-fold>


    def test(self, test_sentences):
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
                if (i, deps) in our_tree_dict:
                    self.total_edges_right += 1
                self.total_edges_checked += 1

                # <editor-fold desc="Getters & Setters">

                # def getWordFeatureIndex(self, word) -> int:
                #     return self.vocabulary_dic[word]
                #
                # def getWordsCount(self) -> int:
                #     return len(self.vocabulary_dic)

                # def getTagsCount(self):
                #     return len(self.tag_dic)

                # def getWordsWeight(self, w1, w2) -> float:
                #     if w1 not in self.word_weight or w2 not in self.word_weight[w1]:
                #         return 0
                #     return self.word_weight[w1][w2]

                # def setWordsWeight(self, w1, w2, weight) -> None:
                #     if w1 not in self.word_weight:
                #         self.word_weight[w1] = dict()
                #         self.total_word_weight[w1] = dict()

                # if w2 not in self.word_weight[w1]:
                #     self.word_weight[w1][w2] = weight
                #     self.total_word_weight[w1][w2] = weight
                #
                # else:
                #     self.word_weight[w1][w2] += weight

                #   def getTagsWeight(self, t1, t2) -> float:
                #         if t1 not in self.tag_weight or t2 not in self.tag_weight[t1]:
                #             return 0
                # return self.tag_weight[t1][t2]

    # def setTagsWeight(self, t1, t2, weight) -> None:
    #     if t1 not in self.tag_weight:
    #         self.tag_weight[t1] = dict()
    #         self.total_tag_weight[t1] = dict()
    #     if t2 not in self.tag_weight[t1]:
    #         self.tag_weight[t1][t2] = weight
    #         self.total_tag_weight[t1][t2] = weight
    #     else:
    #         self.tag_weight[t1][t2] += weight

    def getDistanceWeight(self, w1, w2, distance) -> float:
        dist = min(distance, 4)
        if w1 not in self.word_distance_weight \
                or w2 not in self.word_distance_weight[w1] \
                or dist not in self.word_distance_weight[w1][w2]:
            return 0
        return self.word_distance_weight[w1][w2][dist]

    def setDistanceWeight(self, w1, w2, distance, weight) -> None:
        dist = min(distance, 4)
        if w1 not in self.word_distance_weight:
            self.word_distance_weight[w1] = dict()
        if w2 not in self.word_distance_weight[w1]:
            self.word_distance_weight[w1][w2] = dict()
        if dist not in self.word_distance_weight[w1][w2]:
            self.word_distance_weight[w1][w2][dist] = weight
        else:
            self.word_distance_weight[w1][w2][dist] += weight

    # </editor-fold>


    # testing
    def getWordsWeightOPT(self, w1, w2) -> float:
        if w1 not in self.map_words_to_vector or w2 not in self.map_words_to_vector[w1]:
            return 0
        return self.feature_weight_vector[self.map_words_to_vector[w1][w2]]

    def setWordsWeightOPT(self, w1, w2, weight,vec) -> None:
        if w1 not in self.map_words_to_vector:
            self.map_words_to_vector[w1] = dict()
        if w2 not in self.map_words_to_vector[w1]:
            self.map_words_to_vector[w1][w2] = self.cur_feature_count
            self.cur_feature_count += 1
            vec = np.hstack((vec,[weight]))
            # self.feature_weight_vector = np.hstack((self.feature_weight_vector, [weight]))
        else:
            # self.feature_weight_vector[self.map_words_to_vector[w1][w2]] += weight
            vec[self.map_words_to_vector[w1][w2]] += weight
        return vec

    def getTagsWeightOPT(self, t1, t2) -> float:
        if t1 not in self.map_tags_to_vector or t2 not in self.map_tags_to_vector[t1]:
            return 0
        return self.feature_weight_vector[self.map_tags_to_vector[t1][t2]]

    def setTagsWeightOPT(self, t1, t2, weight,vec) -> None:
        if t1 not in self.map_tags_to_vector:
            self.map_tags_to_vector[t1] = dict()
        if t2 not in self.map_tags_to_vector[t1]:
            self.map_tags_to_vector[t1][t2] = self.cur_feature_count
            self.cur_feature_count += 1
            vec = np.hstack((vec,[weight]))
            # self.feature_weight_vector = np.hstack((self.feature_weight_vector, [weight]))
        else:
            vec[self.map_tags_to_vector[t1][t2]] += weight
            # self.feature_weight_vector[self.map_tags_to_vector[t1][t2]] += weight
        return vec




    def OmerDontDeleteMeOmerDontDeleteMeOmerDontDeleteMeOmerDontDeleteMeOmerDontDeleteMeOmerDontDeleteMe(self):
        pass
