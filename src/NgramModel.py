from .Tokenizer import Tokenizer


class NGModel:
    def __init__(self, file_name: str, name: str, orders: int = 1):
        self.token = Tokenizer(file_name)
        self.orders = orders
        self.name = name
        self.ngrams = self.__get_probs()

    def __ngram(self, tokens, order):
        counters = {}
        L = len(tokens) - order
        for i in range(L):
            current = "".join(tokens[i * order:(i + 1) * order])
            if counters.get(current):
                counters[current] += 1
            else:
                counters[current] = 1
        return counters

    def __get_probs(self):
        ngrams = {}
        token_list = list(self.token)
        for order in range(1, self.orders + 1):
            ngrams[order] = self.__ngram(token_list, order)
        return ngrams

    def __repr__(self):
        return self.name
