from ex4.IterativeExtractor import IterativeExtractor
from ex4.TreeExtractor import TreeExtractor

import random

def test_page(page_name):
    # ie = IterativeExtractor(page_name)
    ie = TreeExtractor(page_name)
    triplets = ie.process()
    test = triplets[:]
    random.shuffle(test)
    real_test = test[:10]
    log = open("log"+page_name.replace(" ",""),mode='w')
    for triple in real_test:
        print("="*25+"Start"+"="*25)
        print(triple)
        log.write(str(triple)+'\n')
        print("="*25+"End"+"="*25)
    print(len(triplets))


def main():
    random.seed(1234)
    page_name = "Donald Trump"
    test_page(page_name)
    page_name = "Brad Pitt"
    test_page(page_name)
    page_name = "Angelina Jolie"
    test_page(page_name)

if __name__ == '__main__':
    main()
