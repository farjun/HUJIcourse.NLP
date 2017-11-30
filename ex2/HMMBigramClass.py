import numpy as np


class HMMBigramTagger:
    # <editor-fold desc="Init">

    def __init__(self, pseudo_words_to_tag=None, delta=0):
        self.__word_to_tag_count = {}
        self.__tag_to_next_tag_count = {}
        self.__tags_count = {}
        self.__pseudo_words_to_tag = pseudo_words_to_tag
        self.__delta = delta

    # </editor-fold>

    # <editor-fold desc="Train">

    def train(self, train_sentences) -> None:
        for sentence in train_sentences:
            for i in range(-1, len(sentence) - 1):
                if i == -1:
                    word, tag = "START", '*'
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
        self.__tag(sentence=test_sentences[0])
        return 0

    def __tag(self, sentence:list):
        # Print the sentence
        for w, t in sentence:
            print(w, end='\t')
        print()

        n = len(sentence)
        Sk = []
        list_of_possible_tags = self.__getPossibleTags()
        # set Sk (tags)
        Sk.append(["*"])
        for i in range(1, n):
            Sk.append(list_of_possible_tags[:])

        a = [("Start","*")]
        probMatrix = self.setPorbMatrix(Sk, sentence)

    def __findMaxProbabilityFromLastRow(self, probability_matrix_row, word, possible_prev_tags, cur_tag):
        tag_probabilities = []
        # print("word : ",word)
        # print("cur_tag : ", cur_tag)
        emission = self.__getEmission(cur_tag, word)
        for j in range(len(possible_prev_tags)):
            perv_tag = possible_prev_tags[j]
            q = self.__getQProbability(cur_tag, perv_tag)
            pi = probability_matrix_row[j]
            tag_probabilities.append(q * emission * pi)
            # print("probability of them: ", q*emission*pi)
        return max(tag_probabilities)

    def __getQProbability(self, cur_tag, perv_tag):
        if cur_tag not in self.__tags_count:
            return 0
        if perv_tag in self.__tag_to_next_tag_count:
            if cur_tag in self.__tag_to_next_tag_count[perv_tag]:
                return self.__tag_to_next_tag_count[perv_tag][cur_tag] / self.__tags_count[cur_tag]
        return 0

    def setPorbMatrix(self, Sk, sentence):
        sentence_length = len(sentence)
        number_of_tags = len(Sk[1])
        # Todo added a "+1" due to we start from "*" and not fro, the first word.
        probabilityMatrix = np.zeros((sentence_length + 1, number_of_tags))

        # set probability of Start to 1 in all rows
        probabilityMatrix[0] = 1
        for i in range(1, sentence_length):
            word = sentence[i][0]
            probability_matrix_row = probabilityMatrix[i - 1]
            possible_prev_tags = Sk[i - 1]
            for j in range(number_of_tags):
                probabilityMatrix[i, j] = self.__findMaxProbabilityFromLastRow(probability_matrix_row, word,
                                                                               possible_prev_tags, Sk[i][j])
        return probabilityMatrix

    def __getPossibleTags(self) -> list:
        return list(self.__tags_count.keys())

    def __getEmission(self, tag, word) -> float:
        print(word)
        if word not in self.__word_to_tag_count:
            # print("didn't saw {word} in training.".format(word=word))
            return 0
        if tag not in self.__word_to_tag_count[word]:
            # print("didn't saw {word} with {tag} in training.".format(word=word,tag=tag))
            return 0
        appearance = self.__word_to_tag_count[word][tag] + self.__delta
        totalTag = self.__tags_count[tag] + len(self.__word_to_tag_count) * self.__delta
        return appearance / totalTag

    # </editor-fold>

    def aaa(self):
        pass
