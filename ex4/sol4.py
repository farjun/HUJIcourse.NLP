from ex4.IterativeExtractor import IterativeExtractor
from ex4.TreeExtractor import TreeExtractor

if __name__ == '__main__':
    ie = IterativeExtractor("Donald Trump")
    triplets = ie.process()
    print(len(triplets))
    ie = IterativeExtractor("Brad Pitt")
    triplets = ie.process()
    print(len(triplets))
    ie = IterativeExtractor("Angelina Jolie")
    triplets = ie.process()
    print(len(triplets))

    #
