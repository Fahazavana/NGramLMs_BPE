from .Tokenizer import Tokenizer
import numpy as np
from .utils import sort_dict


class NGModel:
    def __init__(self, file_name: str, vocab:list, name: str, orders: int = 1):
        self.vocab = vocab
        self.token = Tokenizer(file_name)
        self.orders = orders
        self.name = name
        self.log_joints = self.__get_probs()

    def __ngram(self, tokens, order):
        counters = {}
        L = len(tokens) - order
        for i in range(L):
            current = tuple(tokens[i * order:(i + 1) * order])
            if counters.get(current):
                counters[current] += 1
            else:
                counters[current] = 1
        return counters

    def __get_probs(self):
        ngrams = {}
        token_list = list(self.token)
        for order in range(1, self.orders + 1):
            _tmp = self.__ngram(token_list, order)
            w = sum(_tmp.values())
            ngrams[order] = sort_dict({k: np.log(v/w)
                                      for k, v in _tmp.items()})
        return ngrams

    def generate(self, start, max_len=100):
        text = ""
        tokens = ["<s>", start]
        for _ in range(max_len):
            tokens = tokens[-(self.orders -2):]
            probs = self.__get_next_gram_probs(tokens)
            next_word = self.__sample_words(probs)
            tokens.append(next_word)
            text += ''.join(next_word)
        return text


    def __get_next_gram_probs(self, tokens):
        order = self.orders
        context = tokens[-(self.orders -2):]
        probs = []
        for i in range(len(self.vocab)):
            joint = context + [self.vocab[i]]
            num = np.exp(self.log_joints[self.orders].get(joint))
            den = np.exp(self.log_joints[self.orders -1].get(context))
            probs[i] = num/den
        return vocab
    
    def __sample_word(self, probs):
        idx = np.random.multinomial(1, probs).argmax() 
        return self.vocab[idx]
    
    def __repr__(self):
        return self.name
