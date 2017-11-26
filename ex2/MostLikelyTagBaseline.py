class MostLikelyTagBaseline:
    def __init__(self):
        self._words_tag_count = dict()
        pass

    def train(self, train_sentences):
        for sentence in train_sentences:
            for word, tag in sentence:
                if word not in self._words_tag_count:
                    self._words_tag_count[word] = dict()
                if tag not in self._words_tag_count[word]:
                    self._words_tag_count[word][tag] = 1
                else:
                    self._words_tag_count[word][tag] += 1

    def _getBestTag(self, word):
        if word not in self._words_tag_count:
            return "NN"
        tagsMap = self._words_tag_count[word]
        bestTag = "NN"
        bestCount = 0
        for tag, count in tagsMap.items():
            if count > bestCount:
                bestTag = tag
                bestCount = count
        return bestTag

    def test(self, test_sentences):
        wrong = 0
        right = 0
        for sentence in test_sentences:
            for word, actualTag in sentence:
                expectedTag = self._getBestTag(word)
                if expectedTag == actualTag:
                    right += 1
                else:
                    wrong += 1
        return (wrong / (right + wrong))
