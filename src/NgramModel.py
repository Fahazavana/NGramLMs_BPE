from collections import defaultdict
from typing import List, Tuple, Dict

import numpy as np

from src.utils import sort_dict


class NGModel:
    def __init__(self, name: str, orders: int, tokenizer):
        self.name = name
        self.orders = orders
        self.tokenizer = tokenizer
        self.counters = {}
        self.vocab = []
        self.vocab_size = 0

    def train(self, file_name):
        print(f"Training {self.name} model...", end=" ")
        for order in range(1, self.orders + 1):
            _tmp = defaultdict(int)
            with open(file_name, "r") as corpus:
                for sentence in corpus:
                    tokenized_sentence = self.tokenizer.charSentenceTokenizer(sentence, order)
                    for ngram in tokenized_sentence:
                        _tmp[ngram] += 1
            self.counters[order] = sort_dict(_tmp)
        self.vocab = [t[0] for t in self.counters[1].keys()]
        self.vocab_size = len(self.vocab)
        print("DONE!")

    def rawcount(self, seq):
        order = len(seq)
        if order > self.orders:
            raise RuntimeError("len(seq) must be less or equal to order")
        if order < 1:
            raise RuntimeError("len(seq) must be greater than 1")

        num = self.counters[order].get(seq, 0)
        if order == 1:
            den = sum(self.counters[order].values())
        else:
            den = self.counters[order - 1].get(seq[:-1], 0)
        return num, den

    def prob(self, seq, alpha: int = 1):
        """
            Compute the probability of a N-gram using:
            - alpha = 0: Unsmoothed
            - alpha = 1: Laplace
            - 1<alpha <0: add-alpha
        """
        num, den = self.rawcount(seq)
        prob = (num + alpha) / (den + alpha * self.vocab_size)
        return prob

    def probInt(self, seq: Tuple[str], alphas: List):
        """
            Compute the probability of an n_gram
            usign interpolation
        """
        prob = 0
        for i in range(self.orders):
            ngram = seq[-self.orders + i:]
            num, den = self.rawcount(ngram)
            if num != 0:
                prob += alphas[i] * (num / den)
        return prob

    def logProbInt(self, sentence: str, alphas: List[float]):
        """
            Compute the log probability of an input sentence
            using interpolation.
        """
        logprob, prob = 0, 0
        for ngram in self.tokenizer.charSentenceTokenizer(sentence, self.orders):
            logprob += np.log(self.probInt(ngram, alphas))
        return logprob

    def logProb(self, sentence, alpha=0):
        logprob, prob = 0, 0
        for ngram in self.tokenizer.charSentenceTokenizer(sentence, self.orders):
            prob = self.prob(ngram, alpha)
            if prob > 0:
                logprob += np.log(prob)
            else:
                return -np.inf
        return logprob

    def __perplexity(self, sentence, params: Dict, mode='default'):
        """
            Compute the perplexity of a Sentence
        """
        T = len(sentence.strip()) + 2  # +2 for start and end tokens
        logprob = 0
        if mode == "default":
            logprob = self.logProb(sentence, params['alpha'])
        elif mode == "inter":
            logprob = self.logProbInt(sentence, params['alphas'])
        return logprob, T

    def perplexity(self, sentence, params: Dict, mode='default', doc=False):
        logprob, T = self.__perplexity(sentence, params, mode)
        if doc:
            return logprob, T
        else:
            return np.exp(-logprob / T)

    def __nextChar(self, context, params: Dict, mode='default'):
        probs = np.zeros(self.vocab_size)
        for i in range(len(self.vocab)):
            ngram = context + (self.vocab[i],)
            if mode == "default":
                probs[i] = self.prob(ngram, params['alpha'])
            elif mode == "inter":
                probs[i] = self.probInt(ngram, params['alphas'])
        probs /= np.sum(probs)
        chr_idx = np.random.multinomial(1, probs).argmax()
        return self.vocab[chr_idx]

    def generateSentence(self, params:Dict, start=None, order=None, mode='default'):
        if order is None:
            order = self.orders
        sentence = [self.tokenizer.start]
        if start is not None:
            sentence += list(start)
        while sentence[-1] != self.tokenizer.end:
            context = sentence[-(order - 1):]
            sentence.append(self.__nextChar(tuple(context), params, mode))
        return "".join(sentence)
