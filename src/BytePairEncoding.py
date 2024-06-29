import re
from collections import defaultdict


class BytePairEncoding:
    def __init__(self, name):
        self.name = name
        self.vocab = set()
        self.merge_rule = []
        self.corpus = defaultdict(int)

    def init_corpus(self, corpus):
        for sentence in corpus:
            sentence = sentence.strip()
            for word in sentence.split():
                key = tuple(word + '_')
                self.corpus[key] += 1
            self.vocab.update(re.sub(r'\s+', '_', sentence))

    def init(self, data):
        if isinstance(data, str):
            try:
                with open(data, 'r') as corpus:
                    self.init_corpus(corpus)
            except FileNotFoundError:
                print(f'File {data} not found')
        elif isinstance(data, list):
            self.init_corpus(data)

    def get_stat(self):
        """"
            Compute the frequency of each pair in the corpus.
        """
        pairs = defaultdict(int)
        for word, count in self.corpus.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += count

        return pairs

    def merge(self, best):
        _corpus = []
        new_vocab = ''.join(best)
        _corpus = defaultdict(int)
        for word, count in self.corpus.items():
            tmp = []
            i = 0
            _word = ()
            while i < len(word) - 1:
                if (word[i], word[i + 1]) == best:
                    _word += (new_vocab,)
                    i += 2
                else:
                    _word += (word[i],)
                    i += 1
            if i < len(word):
                _word += (word[i],)
            _corpus[_word] = count
        self.corpus = _corpus
        self.vocab.add(new_vocab)

    def learn(self, max_iter=500):
        print(f'Token learning for: {self.name}')
        for i in range(max_iter):
            frequent_pair = self.get_stat()
            if not frequent_pair:
                print("Stop learning!! No more pairs")
                break
            best = max(frequent_pair, key=frequent_pair.get)
            self.merge(best)
            self.merge_rule.append((best[0], best[1], ''.join(best)))
        print("Encoding Done")

    def decode(self):
        raise NotImplementedError


if __name__ == '__main__':
    corpus = "low low low low low\nlower lower\nnewer newer newer newer newer newer\nwider wider wider\nnew new new"
    corpus = re.split(r'\n', corpus)
    en_bpe = BytePairEncoding('en')
    en_bpe.init(corpus)
    en_bpe.learn(max_iter=100)
