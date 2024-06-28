from collections import defaultdict

import numpy as np

from src.utils import sort_dict


class NGModel:
    def __init__(self, name: str, orders: int, tokenizer):
        self.name = name
        self.orders = orders
        self.tokenizer = tokenizer
        self.counters = {}
        self.vocab = set()
        self.vocab_size = 0

    def train(self, file_name):
        print(f"Training {self.name} model...", end=" ")
        self.vocab = set()
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

    def Prob(self, seq, alpha=1):
        order = len(seq)
        if order > self.orders:
            raise RuntimeError("len(seq) must be less or equal to order")

        if order == 1:
            count = self.counters[order].get(seq, 0)
            total_count = sum(self.counters[order].values())
            prob = (count + alpha) / (total_count + alpha * self.vocab_size)
        else:
            # For higher-order n-grams
            count = self.counters[order].get(seq, 0)
            prefix = seq[:-1]
            prefix_count = self.counters[order - 1].get(prefix, 0)
            prob = (count + alpha) / (prefix_count + alpha * self.vocab_size)
        return prob

    def LogProb(self, sentence, alpha=1):
        logprob = 0
        for ngram in self.tokenizer.charSentenceTokenizer(sentence, self.orders):
            order = len(ngram)
            if order > self.orders:
                raise RuntimeError("len(seq) must be less or equal to order")

            if order == 1:
                # For unigrams
                count = self.counters[order].get(ngram, 0)
                total_count = sum(self.counters[order].values())
                prob = (count + alpha) / (total_count + alpha * self.vocab_size)
            else:
                # For higher-order n-grams
                count = self.counters[order].get(ngram, 0)
                prefix = ngram[:-1]
                prefix_count = self.counters[order - 1].get(prefix, 0)
                prob = (count + alpha) / (prefix_count + alpha * self.vocab_size)

            if prob > 0:
                logprob += np.log(prob)
            else:
                logprob += -np.inf
        return logprob

    def perplexity(self, sentence, alpha=1, doc=False):
        T = len(sentence.strip()) + 2  # +2 for start and end tokens
        logprob = self.LogProb(sentence, alpha)
        if doc:
            return logprob, T
        return np.exp(-logprob / T)

    def __nextChar(self, context, alpha=1):
        probs = np.zeros(self.vocab_size)
        for i in range(len(self.vocab)):
            ngram = context + [self.vocab[i]]
            probs[i] = self.Prob(tuple(ngram), alpha=1)
        probs /= np.sum(probs)
        chr_idx = np.random.multinomial(1, probs).argmax()
        return self.vocab[chr_idx]

    def generateSentence(self, start=None, order=None, alpha=1):
        if order is None:
            order = self.orders
        sentence = [self.tokenizer.start]
        if start is not None:
            sentence += list(start)
        while sentence[-1] != self.tokenizer.end:
            context = sentence[-(order - 1):]
            sentence.append(self.__nextChar(context))
        return "".join(sentence)
