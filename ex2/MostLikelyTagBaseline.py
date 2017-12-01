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

    def test(self, test_sentences):
        wrong = 0
        total = 0
        seen = 0
        seenWrong = 0
        unseen = 0
        unseenWrong = 0
        for sentence in test_sentences:
            for word, actualTag in sentence:
                expectedTag,isUnseen = self._getBestTag(word)
                if expectedTag != actualTag:
                    wrong += 1
                    if isUnseen:
                        unseenWrong += 1
                    else:
                        seenWrong += 1
                if isUnseen:
                    unseen+=1
                else:
                    seen += 1
                total += 1
        return (wrong/total),(seenWrong/seen),(unseenWrong/unseen)

    def _getBestTag(self, word):
        if word not in self._words_tag_count:
            return "NN", True
        tagsMap = self._words_tag_count[word]
        bestTag = max(tagsMap, key=tagsMap.get)
        return bestTag , False
