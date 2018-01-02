import numpy as np
import ex3.MSTAlgorithem as MSTAlgorithem
import nltk


class MSTParser:
    def __init__(self, distance_flag=False):
        # conputes which tuples will have the value 1 in the sentence
        # error vars
        self.total_edges_checked = 0
        self.total_edges_right = 0

        # flags
        self.distance_flag = distance_flag

        # testing
        self.total_feature_weight_vector = np.array([], dtype=np.float64)
        self.feature_weight_vector = np.array([], dtype=np.float64)

        self.map_words_to_vector = {}
        self.cur_feature_count = 0
        self.map_tags_to_vector = {}
        self.cur_tags_count = 0

        # distance feature
        self.distance_weight_vecotr = np.array([0] * 4, dtype=np.float64)
        self.total_distance_weight_vecotr = np.array([0] * 4, dtype=np.float64)

    def train(self, train_sentences):
        random_range = np.arange(train_sentences.size)
        np.random.shuffle(random_range)

        for i in random_range:
            cur_sentence_tree = train_sentences[i]
            mst_graph = self.tag(cur_sentence_tree)
            self.set_new_weights_by_trees(mst_graph, cur_sentence_tree.nodes)

        np.random.shuffle(random_range)
        for i in random_range:
            cur_sentence_tree = train_sentences[i]
            mst_graph = self.tag(cur_sentence_tree)
            self.set_new_weights_by_trees(mst_graph, cur_sentence_tree.nodes)

        # after the training we normelize the data
        self.normelize_total_weight_dict(train_sentences.size)
        self.feature_weight_vector = self.total_feature_weight_vector
        self.distance_weight_vecotr = self.total_distance_weight_vecotr

    def tag(self, cur_sentence_tree):
        full_graph = self.get_full_graph_from_dict(cur_sentence_tree.nodes)
        mst_graph = MSTAlgorithem.min_spanning_arborescence(full_graph, 0)
        return mst_graph

    def get_full_graph_from_dict(self, sentence_dict):
        total_indexes = len(sentence_dict)
        arcs = []
        for fromIndex in range(1, total_indexes):
            word1 = sentence_dict[fromIndex]
            for toIndex in range(total_indexes):
                if fromIndex == toIndex:
                    continue

                word2 = sentence_dict[toIndex]
                cur_weight = self.getWordsWeightOPT(word1['word'], word2['word'])
                cur_weight += self.getTagsWeightOPT(word1['tag'], word2['tag'])
                if self.distance_flag:
                    if word1['address'] < word2['address']:
                        cur_weight += self.getDistanceWeight(word2['address'] - word1['address'])
                arcs.append(MSTAlgorithem.Arc(fromIndex, -1 * cur_weight, toIndex))
        return arcs

    def set_new_weights_by_trees(self, our_tree, real_tree):
        # add arcs of real tree by iterating the nodes (hard)
        for i in range(len(real_tree)):
            cur_word_dict = real_tree[i]
            if cur_word_dict['head'] is not None:
                head_address = cur_word_dict['head']
                self.setWordsWeightOPT(cur_word_dict['word'], real_tree[head_address]['word'], 1)
                self.setTagsWeightOPT(cur_word_dict['tag'], real_tree[head_address]['tag'], 1)
                if self.distance_flag:
                    if cur_word_dict['address'] < head_address:
                        self.setDistanceWeight(head_address - cur_word_dict['address'], 1)

        # add arcs of our tree by iterating the arcs (easy)
        for i in range(1, len(our_tree) + 1):
            from_word_index = our_tree[i].tail
            to_word_index = our_tree[i].head
            self.setWordsWeightOPT(real_tree[from_word_index]['word'], real_tree[to_word_index]['word'], -1)
            self.setTagsWeightOPT(real_tree[from_word_index]['tag'], real_tree[to_word_index]['tag'], -1)
            if self.distance_flag:
                if from_word_index < to_word_index:
                    self.setDistanceWeight(real_tree[to_word_index]['address'] - real_tree[from_word_index]['address'],
                                           -1)

        # sum up the prev vecotrs
        diff = self.feature_weight_vector.size - self.total_feature_weight_vector.size
        if diff:
            self.total_feature_weight_vector = np.hstack(
                (self.total_feature_weight_vector, np.zeros((diff,))))
        self.total_feature_weight_vector += self.feature_weight_vector
        self.total_distance_weight_vecotr += self.distance_weight_vecotr

    def normelize_total_weight_dict(self, N):
        self.total_feature_weight_vector /= (2 * N)
        self.total_distance_weight_vecotr /= (2 * N)

    # </editor-fold>


    def test(self, test_sentences):
        errors = np.zeros((test_sentences.size,),dtype=np.float64)
        i = 0
        for i in range(test_sentences.size):
            cur_sentence_tree = test_sentences[i]
            mst_graph = self.tag(cur_sentence_tree)
            errors[i] = self.calculate_error(mst_graph, cur_sentence_tree.nodes)
            i += 1
        return np.mean(errors)

    def calculate_error(self, our_tree, real_tree):
        our_tree_dict = MSTAlgorithem.turn_output_to_dict(our_tree)
        right = 0
        checked = 0
        for i in range(len(real_tree)):
            cur_word_dict = real_tree[i]
            if cur_word_dict['head'] is not None:
                if (i, cur_word_dict['head']) in our_tree_dict:
                    right += 1
                checked += 1
        return right/checked

    # <editor-fold desc="Getters & Setters">

    def getDistanceWeight(self, distance) -> float:
        dist = min(distance, 4) - 1
        return self.distance_weight_vecotr[dist]

    def setDistanceWeight(self, distance, weight) -> None:
        dist = min(distance, 4) - 1
        self.distance_weight_vecotr[dist] += weight

    # </editor-fold>

    # testing
    def getWordsWeightOPT(self, w1, w2) -> float:
        if w1 not in self.map_words_to_vector or w2 not in self.map_words_to_vector[w1]:
            return 0
        return self.feature_weight_vector[self.map_words_to_vector[w1][w2]]

    def setWordsWeightOPT(self, w1, w2, weight) -> None:
        if w1 not in self.map_words_to_vector:
            self.map_words_to_vector[w1] = dict()

        if w2 not in self.map_words_to_vector[w1]:
            self.map_words_to_vector[w1][w2] = self.cur_feature_count
            self.cur_feature_count += 1
            self.feature_weight_vector = np.hstack((self.feature_weight_vector, [weight]))
        else:
            self.feature_weight_vector[self.map_words_to_vector[w1][w2]] += weight

    def getTagsWeightOPT(self, t1, t2) -> float:
        if t1 not in self.map_tags_to_vector or t2 not in self.map_tags_to_vector[t1]:
            return 0
        return self.feature_weight_vector[self.map_tags_to_vector[t1][t2]]

    def setTagsWeightOPT(self, t1, t2, weight) -> None:
        if t1 not in self.map_tags_to_vector:
            self.map_tags_to_vector[t1] = dict()
        if t2 not in self.map_tags_to_vector[t1]:
            self.map_tags_to_vector[t1][t2] = self.cur_feature_count
            self.cur_feature_count += 1
            self.feature_weight_vector = np.hstack((self.feature_weight_vector, [weight]))
        else:
            self.feature_weight_vector[self.map_tags_to_vector[t1][t2]] += weight

    def OmerDontDeleteMeOmerDontDeleteMeOmerDontDeleteMeOmerDontDeleteMeOmerDontDeleteMeOmerDontDeleteMe(self):
        pass
