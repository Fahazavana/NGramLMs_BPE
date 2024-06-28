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
                    tokenized_sentence = self.tokenizer.charSentenceTokenizer(
                        sentence, order
                    )
                    for ngram in tokenized_sentence:
                        _tmp[ngram] += 1
                        self.vocab.update(ngram)
            self.counters[order] = sort_dict(_tmp)
        self.vocab_size = len(self.vocab)
        print("DONE!")

    def Prob(self, seq, order):
        n = len(seq)
        if n > order:
            raise RuntimeError("len(seq) must be less or equal to order")
        elif n == 1:
            return self.counters[n].get(seq, 0) / sum(self.counters[n].values())
        else:
            pw1t = self.counters[order].get(seq, 0)
            pw1t_1 = self.counters[order - 1].get(seq[: order - 1], 0)
            if pw1t_1 == 0:
                return 0
            return pw1t / pw1t_1

    def LogProb(self, sentence, alpha=1):
        logprob = 0
        counts = 0
        for ngram in self.tokenizer.charSentenceTokenizer(sentence, self.orders):
            order = len(ngram)
            if order > self.orders:
                raise RuntimeError("len(seq) must be less or equal to order")

            if order == 1:
                # For unigrams
                count = self.counters[order].get(ngram, 0)
                total_count = sum(self.counters[order].values())
                prob = (count + alpha) / \
                    (total_count + alpha * self.vocab_size)
            else:
                # For higher-order n-grams
                count = self.counters[order].get(ngram, 0)
                prefix = ngram[:-1]
                prefix_count = self.counters[order - 1].get(prefix, 0)
                prob = (count + alpha) / \
                    (prefix_count + alpha * self.vocab_size)

            if prob > 0:
                logprob += np.log(prob)
            else:
                logprob += -np.inf
            counts += 1
        return logprob, counts

    def perplexity(self, sentence, alpha=1, doc=False):
        """
        """
        T = len(sentence.strip()) + 2  # +2 for start and end tokens
        logprob = self.LogProb(sentence, alpha)[0]
        if doc:
            return logprob, T
        else:
            return np.exp(-logprob / T)
