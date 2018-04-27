

corpus_name = "corpus_ex1"
freq_list_name = "freq_list_ex1"
if __name__ == '__main__':
    corpus = open(corpus_name)
    lines = [line for line in corpus]
    print(len(lines))
    freq_list = open(freq_list_name)
    lines = [line for line in freq_list]
    print(len(lines))