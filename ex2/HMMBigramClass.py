import numpy as np


class HMMBigramTagger:
    def __init__(self, pseudo_words_to_tag=None):
        self._word_to_tag_count = {}
        self._tag_to_next_tag_count = {}
        self._tags_count = {}
        self._pseudo_words_to_tag = pseudo_words_to_tag

    def train(self, train_sentences):
        for sentence in train_sentences:
            for i in range(-1, len(sentence) - 1):
                if i == -1:
                    word, tag = "START", '*'
                else:
                    word, tag = sentence[i]
                _, next_tag = sentence[i + 1]
                self._updateWordToTag(tag=tag, word=word)
                self._updateTagToNextTag(tag=tag, next_tag=next_tag)
                self._updateTagCount(tag=tag)

    def test(self, test_sentences) -> float:
        pass

    def _updateTagCount(self, tag):
        if tag not in self._tags_count:
            self._tags_count[tag] = 1
        else:
            self._tags_count[tag] += 1

    def _updateTagToNextTag(self, tag, next_tag):
        if tag not in self._tag_to_next_tag_count:
            self._tag_to_next_tag_count[tag] = dict()
        if next_tag not in self._tag_to_next_tag_count[tag]:
            self._tag_to_next_tag_count[tag][next_tag] = 1
        else:
            self._tag_to_next_tag_count[tag][next_tag] += 1

    def _updateWordToTag(self, tag, word):
        if word not in self._word_to_tag_count:
            self._word_to_tag_count[word] = dict()
        if tag not in self._word_to_tag_count[word]:
            self._word_to_tag_count[word][tag] = 1
        else:
            self._word_to_tag_count[word][tag] += 1



    def findMaxProbabilityFromLastColum(self, probability_matrix_column, word, possible_prev_tags, cur_tag):
        tag_probabilities = []

        print("word : ",word)
        print("cur_tag : ", cur_tag)

        emission = self._word_to_tag_count[word][cur_tag]

        for j in range(len(possible_prev_tags)):

            prev_tag = possible_prev_tags[j]
            p = self._tag_to_next_tag_count[prev_tag][cur_tag]/ self._tags_count[prev_tag]
            pi = probability_matrix_column[j]

            tag_probabilities.append(p*emission*pi)
        return max(tag_probabilities)

    def setPorbMatrix(self, Sk, sentence):
        sentence_length = len(sentence)
        number_of_tags = len(Sk)
        probabilityMAtrix = np.zeros((number_of_tags, sentence_length))

        # set probability of Start to 1 in all rows
        probabilityMAtrix[:0] = 1
        # set probability of Stop to 1 in all rows
        probabilityMAtrix[:-1] = 1

        for i in range(1,sentence_length):
            for j in range(number_of_tags):
                probabilityMAtrix[i,j] = self.findMaxProbabilityFromLastColum(probabilityMAtrix[i:],sentence[i],Sk[i],Sk[i][j])

        return probabilityMAtrix

    def _getPossibleTags(self):
        return list(self._tags_count.keys())

    def doAdd1Smooth(self):
        possible_tags = self._getPossibleTags()
        for tag_count_map in self._word_to_tag_count.values():
            for tag in possible_tags:
                if tag not in tag_count_map:
                    tag_count_map[tag] = 1
                else:
                    tag_count_map[tag] += 1

    def tag(self, sentence):
        n = len(sentence)
        Sk = []
        list_of_possible_tags = self._getPossibleTags()
        # set Sk (tags)
        Sk.append(["*"])
        for i in range(1, n):
            Sk.append(list_of_possible_tags[:])

        probMatrix = self.setPorbMatrix(Sk, sentence)
        print(probMatrix)
