import re
from collections import defaultdict
from typing import List, Tuple, TextIO, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class BytePairEncoding:
    """
    Token learner with byte-pair encoding
    """
    def __init__(self, name:str):
        self.name = name
        self.vocab = set()
        self.history = []
        self.corpus = defaultdict(int)

    def __initBpe(self, corpus:Union[TextIO, List[str]]):
        """
        Initialize the corpus dictionnary and vocabulary.
            key: words in the corpus splited as characters
            value: frequency of the words in the corpus.
        Args:
            corpus: List of sentence or line from a text file
        Returns:

        """
        for sentence in corpus:
            sentence = sentence.strip()
            for word in sentence.split():
                key = tuple(word + '_')
                self.corpus[key] += 1
            self.vocab.update(re.sub(r'\s+', '_', sentence))

    def init(self, data:Union[List[str], str])->None:
        """
        Initialize the vocabulary and corpus dictionary.
        it will call __initBpe depending on the data type
        Args:
            data: List of sentece or file path
        Returns:

        """
        if isinstance(data, str):
            try:
                with open(data, 'r') as corpus:
                    self.__initBpe(corpus)
            except FileNotFoundError:
                print(f'File {data} not found')
        elif isinstance(data, list):
            self.__initBpe(data)

    def __count_pairs(self):
        """"
            Compute the frequency of each pair in the corpus.
        """
        pairs = defaultdict(int)
        for word, count in self.corpus.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += count
        return pairs

    def __merge(self, best:Tuple[str,str]):
        """
            Merge the most frequent pairs in the corpus (key of self.corpus)
            After the merge it updates the vocabulary and corpus.
        Args:
            best: The most frequent pair in the corpus.

        Returns:

        """
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

    def learn(self, max_iter:int=500):
        """
        Learn "max_iter" new type from the corpus, when there is no more pairs, all the key of 'self.corpus' have been merged.
        it will stop.
        It also records the merging history, which can be to tokenize new corpus.
        Args:
            max_iter: maximum number of iteration wich is the maximum number of merge.

        Returns:

        """
        print(f'Token learning for: {self.name}')
        for i in range(max_iter):
            frequent_pair = self.__count_pairs()
            if not frequent_pair:
                print("Stop learning!! No more pairs")
                break
            best = max(frequent_pair, key=frequent_pair.get)
            self.__merge(best)
            self.history.append((best[0], best[1], ''.join(best)))
        print("Encoding Done")

    def token_segmenter(self):
        """
        Tokenize new corpus by applying greedily the merging history
        Returns:
        """
        raise NotImplementedError


def plot_intersection_cmf(cmf: np.ndarray, labels: List[str], title: str = '', figsize: Tuple[int, int] = (6, 5),
                          fmt='.0f') -> None:
    """"
        Plot intersection confusion matrix.
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cmf, annot=True, fmt=fmt)
    plt.xticks(np.arange(len(labels)) + 0.5, labels)
    plt.yticks(np.arange(len(labels)) + 0.5, labels)
    plt.title(fr"{title}")
