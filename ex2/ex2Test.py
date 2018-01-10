import ex2Sol as ex2Sol
# import ex2.ex2Sol as ex2Sol
from HMMBigramClass import HMMBigramTagger as hmm
# from ex2.HMMBigramClass import HMMBigramTagger as hmm

PSEUDO = {}


def generatePseudo(train_sentences, test_sentences):
    m = {}
    for sentence in train_sentences:
        for word, tag in sentence:
            if word not in m:
                m[word] = {}
            if tag in m[word]:
                m[word][tag] += 1
            else:
                m[word][tag] = 1
    for sentence in test_sentences:
        for word, tag in sentence:
            if word not in m:
                m[word] = dict()
            if tag in m[word]:
                m[word][tag] += 1
            else:
                m[word][tag] = 1

    pseudo = {}
    for word in m.keys():
        if sum(m[word].values()) > 5:
            continue
        word = str(word)
        if word.startswith("anti"):
            pseudo[word] = 'against'
        elif word.startswith("pre"):
            pseudo[word] = 'prefix'
        elif word.startswith("mis"):
            pseudo[word] = 'misfire'
        elif word.startswith('over'):
            pseudo[word] = 'over'
        elif word[0].isupper():
            pseudo[word] = 'upper'
        elif any(char.isdigit() for char in word):
            pseudo[word] = 'hasDigit'
        elif word.endswith('ed'):
            pseudo[word] = 'past'
        elif word.endswith('ment'):
            pseudo[word] = 'action'
        elif word.endswith('ly'):
            pseudo[word] = 'charof'
        elif '-' in word:
            pseudo[word] = 'has-'
        elif word.endswith('ing'):
            pseudo[word] = 'ing'
        elif word.endswith('\'s'):
            pseudo[word] = 'owns'
        elif word.endswith('ous'):
            pseudo[word] = 'ous'
        # else:
        #     print("Passed: " + word )
    return pseudo


def part1(train_sentences, test_sentences):
    print("=" * 30)
    print("================ No Pseudo ====================")
    print("=" * 30)
    no_pseudo_error = ex2Sol.optimizedTest(train_sentences, test_sentences, getMatrix=False)
    total, seen, unseen = no_pseudo_error[0]
    print("loss - normal :\n"
          "total:{total}  seen:{seen} unseen:{unseen}"
          .format(total=total, seen=seen, unseen=unseen))
    total, seen, unseen = no_pseudo_error[1]
    print("loss - Smooth :\n"
          "total:{total}  seen:{seen} unseen:{unseen}"
          .format(total=total, seen=seen, unseen=unseen))


def part2(train_sentences, test_sentences):
    pseudo_words = generatePseudo(train_sentences, test_sentences)
    print("mapping")
    for j in range(len(train_sentences)):
        for i in range(len(train_sentences[j])):
            if train_sentences[j][i][0] in pseudo_words:
                train_sentences[j][i][0] = pseudo_words[train_sentences[j][i][0]]
    for j in range(len(test_sentences)):
        for i in range(len(test_sentences[j])):
            if test_sentences[j][i][0] in pseudo_words:
                test_sentences[j][i][0] = pseudo_words[test_sentences[j][i][0]]
    print("done")
    print("=" * 30)
    print("================ Pseudo ====================")
    print("=" * 30)
    with_pseudo_error = ex2Sol.optimizedTest(train_sentences, test_sentences, getMatrix=True)
    total, seen, unseen = with_pseudo_error[0]
    print("loss - pseudo:\n"
          "total:{total}  seen:{seen} unseen:{unseen}"
          .format(total=total, seen=seen, unseen=unseen))
    total, seen, unseen = with_pseudo_error[1]
    print("loss - pseudo and smooth:\n"
          "total:{total}  seen:{seen} unseen:{unseen}"
          .format(total=total, seen=seen, unseen=unseen))
    matrix = with_pseudo_error[2]
    tags = with_pseudo_error[3]
    print("=" * 30)
    print("================ Matrix ====================")
    print("=" * 30)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            print(matrix[i, j], end='\t')
        print()
    print("=" * 30)
    print("================ Tags ====================")
    print("=" * 30)
    for i in range(len(tags)):
        print(tags[i],end= "\t")
    print()


if __name__ == '__main__':

    train_sentences, test_sentences = ex2Sol.getTaggedSentences()
    part1(train_sentences, test_sentences)
    part2(train_sentences, test_sentences)
    total,seen , unseen = ex2Sol.base(train_sentences,test_sentences)
    print("loss - base :\n"
          "total:{total}  seen:{seen} unseen:{unseen}"
          .format(total=total, seen=seen, unseen=unseen))
