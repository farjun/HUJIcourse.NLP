import spacy
import wikipedia

import spacy.tokens
from spacy.tokens import Token

class IterativeExtractor:
    def __init__(self,page_name:str):
        nlp_model = spacy.load('en')
        page = wikipedia.page(page_name).content
        analyzed_page = nlp_model(page)
        self.analyzed_page = analyzed_page

    def process(self) -> list:
        all_propn = self.findAllPropn()
        all_pairs = self.findAllPropnPairs(all_propn)
        all_triplets = self.findAllTriplets(all_pairs)
        return all_triplets

    def findAllPropn(self)->list:
        ret = []
        to_add = []
        for token in self.analyzed_page:
            if token.pos_ == "PROPN":
                to_add.append(token)
            else:
                if to_add:
                    ret.append(to_add)
                    to_add = []
        return ret

    def findAllPropnPairs(self, all_propn:list)->list:
        length = len(all_propn)
        pairs = []
        for i in range(length):
            for j in range(i+1,length):
                if self.toAddPair(all_propn[i][-1], all_propn[j][0]):
                    pairs.append([all_propn[i], all_propn[j]])
        return pairs

    def toAddPair(self, t1:Token, t2:Token)->bool:
        span = self.analyzed_page[t1.i+1:t2.i]
        for t in span:
            if t.pos_ == "PUNCT":
                return False
        for t in span:
            if t.pos_ == "VERB":
                return True
        return False

    def findAllTriplets(self, all_pairs:list)->list:
        triplets = []
        for pair in all_pairs:
            triple = self.getTriple(pair[0],pair[1])
            triplets.append(triple)
        return triplets

    def getTriple(self, t1list:list, t2list:list):
        t1 = t1list[-1]
        t2 = t2list[0]
        span = self.analyzed_page[t1.i+1:t2.i]
        relation = list(filter(lambda token: token.pos_ != "VERB" or token.pos_!="ADP",span))
        return [t1list,relation,t2list]

