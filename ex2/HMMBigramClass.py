import numpy as np


class HMMBigramTagger:
    # <editor-fold desc="Init">

    def __init__(self, pseudo_words_to_tag=None, delta=0):
        self.__word_to_tag_count = {}
        self.__tag_to_next_tag_count = {}
        self.__tags_count = {}
        self.__pseudo_words_to_tag = pseudo_words_to_tag
        self.__delta = delta

        # error variables
        self.__words_counter = 0
        self.__correct_words_counter = 0

    # </editor-fold>

    # <editor-fold desc="Train">

    def train(self, train_sentences) -> None:
        for sentence in train_sentences:
            for i in range(-1, len(sentence)):
                if i == -1:
                    word, tag = "START", '*'
                    _, next_tag = sentence[i + 1]
                elif i == len(sentence) - 1:
                    word, tag = sentence[i]
                    _, next_tag = "END", '*'
                else:
                    word, tag = sentence[i]
                    _, next_tag = sentence[i + 1]
                self.__updateWordToTag(tag=tag, word=word)
                self.__updateTagToNextTag(tag=tag, next_tag=next_tag)
                self.__updateTagCount(tag=tag)

    def __updateTagCount(self, tag) -> None:
        if tag not in self.__tags_count:
            self.__tags_count[tag] = 1
        else:
            self.__tags_count[tag] += 1

    def __updateTagToNextTag(self, tag, next_tag) -> None:
        if tag not in self.__tag_to_next_tag_count:
            self.__tag_to_next_tag_count[tag] = dict()
        if next_tag not in self.__tag_to_next_tag_count[tag]:
            self.__tag_to_next_tag_count[tag][next_tag] = 1
        else:
            self.__tag_to_next_tag_count[tag][next_tag] += 1

    def __updateWordToTag(self, tag, word) -> None:
        if word not in self.__word_to_tag_count:
            self.__word_to_tag_count[word] = dict()
        if tag not in self.__word_to_tag_count[word]:
            self.__word_to_tag_count[word][tag] = 1
        else:
            self.__word_to_tag_count[word][tag] += 1

    # </editor-fold>


    # <editor-fold desc="Test">
    def test(self, test_sentences) -> float:
        self.__generateSK()
        print(len(test_sentences))
        i = 0
        for cur_sentence in test_sentences:
            i += 1
            if i % 100 == 0:
                print("Test Iter: {i}".format(i=i))
            sentence = list(map(lambda var: var[0], cur_sentence))  # remove tags
            tagged_sentence = self.__tag(sentence=sentence)
            correct_tags = list(map(lambda var: var[1], cur_sentence))  # remove words
            self._computeError(tagged_sentence, correct_tags)
        return self.__correct_words_counter / self.__words_counter

    def __generateSK(self):
        self.__sk = [["*"]] + [self.__getPossibleTags()]

    def __getPossibleTags(self) -> list:
        return list(self.__tags_count.keys())

    def _computeError(self, out_tags, correct_tags):
        self.__words_counter += len(out_tags)
        self.__correct_words_counter += \
            np.sum(np.array(out_tags, dtype=np.str) == np.array(correct_tags, dtype=np.str))

    def __tag(self, sentence: list) -> list:
        # set Sk (tags)
        Sk = self.__sk
        list_of_possible_tags = Sk[1]

        # add start to the sentence
        probMatrix, backPointers = self.getProbMatrix(Sk, [("Start", "*")] + sentence)
        bestProbIndex = -1
        bestProb = 0
        for i in range(len(probMatrix[-1])):
            prob = probMatrix[-1][i]
            if prob > bestProb:
                bestProb = prob
                bestProbIndex = i
        tags = self.__constructTags(backPointers, bestProbIndex, list_of_possible_tags, sentence)
        return tags

    def __constructTags(self, backPointers, bestProbIndex, list_of_possible_tags, sentence):
        tags = []
        index = bestProbIndex
        for i in range(len(sentence), 0, -1):
            tag = list_of_possible_tags[index]
            tags = [tag] + tags
            index = backPointers[i][index]
        return tags

    def __findMaxProbabilityFromLastRow(self, probability_matrix_row, word, possible_prev_tags, cur_tag) \
            -> [float, str]:

        bestPrevTagIndex = 0
        bestProbability = 0
        emission = self.__getEmission(cur_tag, word)
        if not emission:
            return bestProbability, bestPrevTagIndex

        if not len(possible_prev_tags) == 1:
            temp = np.array(list(map(lambda prev_tag : self.__getQProbability(cur_tag, prev_tag),possible_prev_tags)),dtype=np.float64)
            temp = emission*temp*probability_matrix_row
            bestProbability = np.max(temp)
            bestPrevTagIndex = (np.nonzero(temp == bestProbability))[0][0]
            return bestProbability,bestPrevTagIndex



        for j in range(len(possible_prev_tags)):
            perv_tag = possible_prev_tags[j]
            q = self.__getQProbability(cur_tag, perv_tag)
            pi = probability_matrix_row[j]
            probability = pi * q * emission

            # print("probability of them: ", q*emission*pi)
            if probability > bestProbability:
                bestProbability = probability
                bestPrevTagIndex = j

        return bestProbability, bestPrevTagIndex

    def __getQProbability(self, cur_tag, prev_tag):
        if cur_tag not in self.__tags_count:
            return 0
        if prev_tag in self.__tag_to_next_tag_count:
            if cur_tag in self.__tag_to_next_tag_count[prev_tag]:
                return self.__tag_to_next_tag_count[prev_tag][cur_tag] / self.__tags_count[cur_tag]
        return 0

    def __getEmission(self, tag, word) -> float:
        if word not in self.__word_to_tag_count:
            # print("didn't saw {word} in training.".format(word=word))
            return 0
        if tag not in self.__word_to_tag_count[word]:
            # print("didn't saw {word} with {tag} in training.".format(word=word,tag=tag))
            return 0
        appearance = self.__word_to_tag_count[word][tag] + self.__delta
        totalTag = self.__tags_count[tag] + len(self.__word_to_tag_count) * self.__delta
        return appearance / totalTag

    def getProbMatrix(self, Sk, sentence) -> [np.ndarray, np.ndarray]:
        sentence_length = len(sentence)
        number_of_tags = len(Sk[1])
        probabilityMatrix = np.zeros((sentence_length, number_of_tags), dtype=np.float64)
        backPointersIndexes = np.zeros((sentence_length, number_of_tags), dtype=np.int)

        # set probability of Start to 1 in all rows
        probabilityMatrix[0] = 1
        backPointersIndexes[0] = 0
        possible_prev_tags = Sk[0]
        tags = Sk[1]
        for i in range(1, sentence_length):
            word = sentence[i]
            probability_matrix_row = probabilityMatrix[i - 1]
            for j in range(number_of_tags):
                probabilityMatrix[i, j], backPointersIndexes[i, j] = \
                    self.__findMaxProbabilityFromLastRow(probability_matrix_row, word, possible_prev_tags, tags[j])
            possible_prev_tags = tags
        return probabilityMatrix, backPointersIndexes

    # </editor-fold>

    def aaa(self):

        pass
