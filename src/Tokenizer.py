class Tokenizer:
    def __init__(self, start="<", end=">"):
        self.start = start
        self.end = end
        
    def charSentenceTokenizer(self, sentence, order):
        sentence = [self.start] + list(sentence.strip()) + [self.end]
        return self.__gen__(sentence, order)

    def __gen__(self, tokens, order):
        if order <= 0:
            raise ValueError("Order must be greater than 0")
        nbr_ngram = len(tokens) - order + 1
        for i in range(nbr_ngram):
            ngram = tuple(tokens[i : i + order])
            if ngram:
                yield ngram

    def wordTokenizer(self, sentence):
        """Tokenize a sentence into words."""
        return sentence.strip().split()