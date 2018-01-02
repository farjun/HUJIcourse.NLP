import numpy as np


class HMMBigramTagger:
    # <editor-fold desc="Init">

    def __init__(self, delta=0, compute_confusion_matrix=0):
        self.__tag_to_index = {}
        self.__wrong_words_unseen = 0
        self.__unseen_words = 0
        self.__wrong_words_seen = 0
        self.__seen_words = 0
        self.__word_to_tag_count = {}
        self.__tag_to_next_tag_count = {}
        self.__tags_count = {}

        #  flags
        self.__delta = delta
        self.__compute_confusion_matrix = compute_confusion_matrix

        # error variables
        self.__words_counter = 0
        self.__wrong_words_counter = 0
        self.__most_common_tag = "."
        self.__bestIndex = 0
        self.__sk = [[]]

        self.__confusion_matrix = []

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
                self.updateWordToTag(tag=tag, word=word)
                self.updateTagToNextTag(tag=tag, next_tag=next_tag)
                self.updateTagCount(tag=tag)
        self.__sk = [["*"]] + [list(self.__tags_count.keys())]
        for i in range(len(self.__sk[1])):
            self.__tag_to_index[self.__sk[1][i]] = i
        self.__most_common_tag = max(self.__tags_count, key=self.__tags_count.get)
        self.__bestIndex = self.__sk[1].index(self.__most_common_tag)

    def updateTagCount(self, tag) -> None:
        if tag not in self.__tags_count:
            self.__tags_count[tag] = 1
        else:
            self.__tags_count[tag] += 1

    def updateTagToNextTag(self, tag, next_tag) -> None:
        if tag not in self.__tag_to_next_tag_count:
            self.__tag_to_next_tag_count[tag] = dict()
        if next_tag not in self.__tag_to_next_tag_count[tag]:
            self.__tag_to_next_tag_count[tag][next_tag] = 1
        else:
            self.__tag_to_next_tag_count[tag][next_tag] += 1

    def updateWordToTag(self, tag, word) -> None:
        if word not in self.__word_to_tag_count:
            self.__word_to_tag_count[word] = dict()
        if tag not in self.__word_to_tag_count[word]:
            self.__word_to_tag_count[word][tag] = 1
        else:
            self.__word_to_tag_count[word][tag] += 1

    # </editor-fold>

    # <editor-fold desc="Test">
    def test(self, test_sentences) -> [float, float, float]:
        if self.__compute_confusion_matrix:
            self.__confusion_matrix = np.zeros((len(self.__sk[1]), len(self.__sk[1])))
        i = 0
        for cur_sentence in test_sentences:
            i += 1
            if i % 100 == 0:
                print("test iter count: {i}".format(i=i))
            sentence = cur_sentence[:, 0]  # remove tags
            correct_tags = cur_sentence[:, 1]  # remove words
            tagged_sentence = self.tag(sentence=sentence)
            self.computeError(tagged_sentence, correct_tags, sentence=sentence)
            #  update confusion matrix
            if self.__compute_confusion_matrix:
                self.updateConfusoinMatrix(tagged_sentence, correct_tags)

        return (self.__wrong_words_counter / self.__words_counter), \
               (self.__wrong_words_seen / self.__seen_words), \
               (self.__wrong_words_unseen / self.__unseen_words)

    def computeError(self, out_tags: np.ndarray, correct_tags: np.ndarray, sentence: np.ndarray) -> None:
        self.__words_counter += len(out_tags)
        self.__wrong_words_counter += np.sum(out_tags != correct_tags)
        seen = np.array([w in self.__word_to_tag_count for w in sentence], dtype=np.bool)
        unseen = ~seen

        self.__seen_words += np.sum(seen)
        self.__wrong_words_seen += np.sum(out_tags[seen] != correct_tags[seen])
        self.__unseen_words += np.sum(unseen)
        self.__wrong_words_unseen += np.sum(out_tags[unseen] != correct_tags[unseen])

    def tag(self, sentence: np.ndarray) -> np.ndarray:
        # set Sk (tags)
        Sk = self.__sk
        list_of_possible_tags = Sk[1]

        # add start to the sentence
        probMatrix, backPointers = self.getProbMatrix(Sk, np.insert(sentence, 0, "START"))

        # for i in range(len(probMatrix[-1])):
        #     probMatrix[-1][i] =  probMatrix[-1][i]*self.getQProbability(list_of_possible_tags[i],"*")
        give_me_name = np.array([self.getQProbability(tag, "*") for tag in list_of_possible_tags])
        bestProbIndex = np.argmax(np.dot(probMatrix[-1], give_me_name), axis=0)

        return self.constructTags(backPointers, bestProbIndex, list_of_possible_tags, sentence)

    def constructTags(self, backPointers, bestProbIndex, list_of_possible_tags, sentence) -> np.ndarray:
        tags = []
        index = bestProbIndex
        for i in range(len(sentence), 0, -1):
            tag = list_of_possible_tags[index]
            tags = [tag] + tags
            index = backPointers[i][index]
        return np.array(tags, dtype=np.str)

    def getProbMatrix(self, Sk, sentence) -> [np.ndarray, np.ndarray]:
        sentence_length = len(sentence)
        number_of_tags = len(Sk[1])
        probabilityMatrix = np.zeros((sentence_length, number_of_tags), dtype=np.float64)
        backPointersIndexes = np.zeros((sentence_length, number_of_tags), dtype=np.int)

        # set probability of Start to 1 in all rows
        probabilityMatrix[0][0] = 1
        backPointersIndexes[0] = 0
        possible_prev_tags = Sk[0]
        tags = Sk[1]
        for i in range(1, sentence_length):
            word = sentence[i]
            probability_matrix_row = probabilityMatrix[i - 1]

            for j in range(number_of_tags):
                probabilityMatrix[i, j], backPointersIndexes[i, j] = \
                    self.findMaxProbabilityFromLastRow(probability_matrix_row, word, possible_prev_tags, tags[j])
            possible_prev_tags = tags
        return probabilityMatrix, backPointersIndexes

    def findMaxProbabilityFromLastRow(self, probability_row, word, prev_tags, cur_tag) -> [float, str]:
        emission = self.getEmission(cur_tag, word)
        if emission == 0:  # unknown word
            bestProbability = 0
            if not len(prev_tags) == 1:
                bestPrevTagIndex = self.__bestIndex
            else:
                bestPrevTagIndex = 0

        elif not len(prev_tags) == 1:  # known word and not first word
            temp = np.array([self.getQProbability(cur_tag, prev_tag) for prev_tag in prev_tags],
                            dtype=np.float64)
            temp = emission * temp * probability_row
            bestProbability = np.max(temp)
            if bestProbability != 0:
                bestPrevTagIndex = np.argmax(temp,axis=0)
            else:
                bestPrevTagIndex = self.__bestIndex
        else:  # first word
            perv_tag = prev_tags[0]
            q = self.getQProbability(cur_tag, perv_tag)
            pi = probability_row[0]
            bestProbability = pi * q * emission
            bestPrevTagIndex = 0

        return bestProbability, bestPrevTagIndex

    def getEmission(self, tag, word) -> float:
        if word not in self.__word_to_tag_count:
            return (1*self.__delta)/len(self.__word_to_tag_count) # Solve zero division
        if tag not in self.__word_to_tag_count[word]:
            return (1*self.__delta)/len(self.__word_to_tag_count) # Solve zero division
        appearance = self.__word_to_tag_count[word][tag] + self.__delta
        total_tag = self.__tags_count[tag] + len(self.__word_to_tag_count) * self.__delta
        return appearance / total_tag

    def getQProbability(self, cur_tag, prev_tag) -> float:
        if prev_tag in self.__tag_to_next_tag_count and cur_tag in self.__tag_to_next_tag_count[prev_tag]:
            return self.__tag_to_next_tag_count[prev_tag][cur_tag] / self.__tags_count[prev_tag]
        return 0

    # </editor-fold>

    # <editor-fold desc="Utils">
    def getWordToTagCount(self):
        return self.__word_to_tag_count

    def getConfusionMatrix(self):
        return self.__confusion_matrix

    def getTags(self):
        return self.__sk[1]

    def setDelta(self, delta):
        self.__delta = delta

    def updateConfusoinMatrix(self, algorithem_tags, correct_tags):
        for i in range(len(algorithem_tags)):
            algorithem_tag = algorithem_tags[i]
            correct_tag = correct_tags[i]
            if correct_tag in self.__tag_to_index:  # check if in a tags MAP
                self.__confusion_matrix[self.__tag_to_index[correct_tag]][self.__tag_to_index[algorithem_tag]] += 1

    def setFlags(self, delta=0, compute_confusion_matrix=0):
        self.__delta = delta
        self.__compute_confusion_matrix = compute_confusion_matrix
    # </editor-fold>

    def placeHolder(self):
        pass