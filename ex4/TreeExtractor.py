import spacy
import spacy.tokens


class TreeExtractor:
    def __init__(self):
        self.triplets = []

    def extract_all_propn_from_token(self, token, output):
        for child in token.children:
            if child.pos_ == "PROPN" and child.dep_ != "compound":
                output.add(child)
            self.extract_all_propn_from_token(child, output)

    def extract_propn_childs_list_from_propn(self, propn_token_set: set):
        propn_childs_list = list()
        for token in propn_token_set:
            temp_set = list()
            temp_set.append(token)
            temp_set.append(list(token.children))
            propn_childs_list.append(temp_set)

        return propn_childs_list

    def condition1(self, h1: spacy.tokens.Token, h2: spacy.tokens.Token):
        return h1.head == h2.head and h1.dep_ == "nsubj" and h2.dep_ == "dobj"

    def condition2(self, h1: spacy.tokens.Token, h2: spacy.tokens.Token):
        if h2.head is None or h2.head.head is None:
            return False
        return h1.head == h2.head.head and h1.dep_ == "nsubj" and h2.dep_ == "prep" and h2.head.dep_ == "pobj"

    def create_subject_relation_object(self,propn_childs_list):
        output = []
        for i in range(len(propn_childs_list)):
            for j in range(len(propn_childs_list)):
                if i == j:
                    continue

                h1_list = propn_childs_list[i]
                h2_list = propn_childs_list[j]

                if self.condition1(h1_list[0],h2_list[0]):
                    output.append([h1_list[0], h1_list[0].head ,h2_list[0]])

                elif self.condition2(h1_list[0],h2_list[0]):
                    output.append([h1_list[0],(h1_list[0].head,h2_list[0].head.head),h2_list[0]])
        return output



    def extract_propn_tags(self, analyzed_page: spacy.tokens.Doc):
        propn_tokens = set()
        for token in analyzed_page:
            self.extract_all_propn_from_token(token, propn_tokens)

        propn_childs_list = self.extract_propn_childs_list_from_propn(propn_tokens)
        self.triplets = self.create_subject_relation_object(propn_childs_list)
        print(self.triplets)

