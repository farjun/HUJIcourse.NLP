

corpus_name = "corpus_ex1"
freq_list_name = "freq_list_ex1"

WINDOW_SIZE =2


import numpy as np

marked_sentences = dict()
def main():
    corpus = open(corpus_name)
    words = [line for line in corpus]
    words = list(map(lambda w: w.replace("\n", ""), words))
    sentences = []
    seed_words_for_water_tank = "oxygen"
    seed_words_for_vehcle_tank = "army"
    s = ""
    for word in words:
        if "<s>" in word:
            s = ""
            continue
        elif "</s>" in word:
            sentences.append(s)
            continue
        else:
            s += word + " "
    # for a in sentences:
    #     if "tank" in a:
    #         print(a)
    water_seed_window_list = get_window_somthing(sentences, seed_words_for_water_tank)
    vehcle_seed_window_list = get_window_somthing(sentences, seed_words_for_vehcle_tank)
    print(water_seed_window_list)
    print(vehcle_seed_window_list)

    for sentence in water_seed_window_list:
        marked_sentences[sentence] = "Army"
    for sentence in water_seed_window_list:
        marked_sentences[sentence] = "Water"

    tank_sents = get_window_somthing(sentence,"tank")



def get_window_somthing(sentences, word):
    sentence_with_word = []
    sentence_with_word_and_tank = []
    for sentence in sentences:
        sentence_list = sentence.split(" ")
        if word in sentence_list:
            a = np.array(sentence_list)
            b1 = np.where(a == word)[0]
            b1_extend = np.vstack((b1 - 2, b1 - 1, b1, b1 + 1, b1 + 2))
            sentence_with_word.append(a[b1_extend].T)


    for sentence in sentence_with_word:
        if "tank" in sentence:
            sentence_with_word_and_tank.append(sentence)

    return sentence_with_word_and_tank

if __name__ == '__main__':
    main()
    pass

