

corpus_name = "corpus_ex1"
freq_list_name = "freq_list_ex1"
if __name__ == '__main__':
    corpus = open(corpus_name)
    words = [line for line in corpus]
    words = list(map(lambda w:w.replace("\n",""),words))

    s = ""
    for word in words:
        if "<s>" in word:
            s = ""
            continue
        elif "</s>" in word:
            print(s)
            continue
        else:
            s += word + " "