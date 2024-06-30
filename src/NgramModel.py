from collections import defaultdict
from typing import List, Tuple, Dict, Union, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from src.utils import sort_dict
from .Tokenizer import Tokenizer


class NGModel:
    """
        N-Gram Model class:
        - Count all N-gram
        - Probability and LogProbability computing
        - Perplexity computing
        - Smooting Methods: unsmoothed, Laplace, Add-k
    """

    def __init__(self, name: str, orders: int, tokenizer: Tokenizer) -> None:
        self.name = name
        self.orders = orders
        self.tokenizer = tokenizer
        self.counters = {}
        self.vocab: List[str] = []
        self.vocab_size = 0

    def train(self, file_name: str) -> None:
        """
        Train the model by generating all count the required (1 to N)-Ngram
        from the corpus document provided from a file. 
        These count will be stored in dictionnary where the key are the N-gram lenght
        Args:
            file_name : path to the corpus file 
        """
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

    def get_count(self, ngram: Tuple) -> Tuple[int, int]:
        """
        Get the N-gram and (n-1)-gram count from the counter,
        these count will be used to compute the probabilty p(w_t|w_{t-n+1:t-1})
        Args:
            ngram: The N-gram in which we ar interested
        Returns:
            (int, int): the N-gram and (n-1)-gram count
        """

        order = len(ngram)
        if order > self.orders:
            raise RuntimeError("len(ngram) must be less or equal to order")
        if order < 1:
            raise RuntimeError("len(ngram) must be greater than 1")

        num = self.counters[order].get(ngram, 0)
        if order == 1:
            den = sum(self.counters[order].values())
        else:
            den = self.counters[order - 1].get(ngram[:-1], 0)
        return (num, den)

    def prob(self, ngram: Tuple, k: float = 1.0) -> float:
        """
            Compute the probability of a N-gram using:
            - k = 0: Unsmoothed
            - k = 1: Laplace
            - 1< k <0: add-k
        Args:
            ngram: The N-gram
            k: The smoothing parameter by default 1
        Returns:
            (float): The N-gram probability
        """
        num, den = self.get_count(ngram)
        prob = (num + k) / (den + k * self.vocab_size)
        return prob

    def probInt(self, ngram: Tuple[str], lambdas: List) -> float:
        """
            Compute the probability of an n_gram
            usign interpolation
        Args:
            ngram: The N-gram
            lambdas: List of the interpoltion weight, the first one correspond to the highest N-gram order
            and the second one correspond to the next higher N-gram order, to the uni-gram order
        Returns:
            (float): The N-gram probability
        """
        prob = 0
        for i in range(self.orders):
            ngram = ngram[-self.orders + i:]
            num, den = self.get_count(ngram)
            if num != 0:
                prob += lambdas[i] * (num / den)
        return prob

    def logProbInt(self, sentence: str, lambdas: List[float]):
        """
            Compute the log probability of an input sentence
            using interpolation.
        Args:
            sentence: The input sentence
            lambdas: List of the interpolation weight, the first one correspond to the highest N-gram order
        Returns:
            (float): The log probability of the input sentence
        """
        logprob, prob = 0, 0
        for ngram in self.tokenizer.charSentenceTokenizer(sentence, self.orders):
            logprob += np.log(self.probInt(ngram, lambdas))
        return logprob

    def logProb(self, sentence:str, k:float=1.0)->float:
        """
        Compute the log probability of an input sentence using the Add-k smoothing
        Args:
            sentence: input sentence
            k: smoothing parameter, by default 1 (Laplace)

        Returns:
            (float): The log probability of the input sentence
        """
        logprob, prob = 0, 0
        for ngram in self.tokenizer.charSentenceTokenizer(sentence, self.orders):
            prob = self.prob(ngram, k)
            if prob > 0:
                logprob += np.log(prob)
            else:
                return -np.inf
        return logprob

    def __perplexity(self, sentence, params: Dict, mode: str = 'add_k') -> Tuple[float, int]:
        """
            Compute the logProbability of a Sentence according to the mode and parameters,
            and the lengnth of the sentence, these value will be used to compute the perplexity
            of the sentence.
        Args:
            sentence: input sentence
            params: Dictionnary of parameters for the model
            mode: Mode for computing the perplexity, by default 'add_k'
                -default: Add-k smoothing -> the params must have the form:
                        {'k': 0.1}
                -inter: Interpolation smoothing -> the parameter must have the form:
                {'lambdas': [0.1, 0.5, 0.4]}
        Returns:
            (float,int): logProbability of the input sentence and its lenght.
        """
        T = len(sentence.strip()) + 2  # +2 for start and end tokens
        logprob = 0
        if mode == "add_k":
            logprob = self.logProb(sentence, params['k'])
        elif mode == "inter":
            logprob = self.logProbInt(sentence, params['lambdas'])
        return logprob, T

    def perplexity(self, sentence: str, params: Dict, mode='add_k', doc=False) -> Union[float, Tuple[float, int]]:
        """
        Compute the perplexity of a sentence or the Logprobabilty of a Sentence and its length if we say doc=True
        Args:
            sentence: the input sentence
            params: Parameters for the model to compute the probabilty
            mode: smoothing mode, by default 'add_k' (Add-k smoothing)
            doc:
                - False for a single sentecce -> perplexity
                - True for as single sentence in adocuments -> logProbability, sentence lenght

        Returns:
            (float): The perplexity of the input sentence and its lenght.
            or
            (float,int): logProbability of the input sentence and its lenght.
        """
        logprob, T = self.__perplexity(sentence, params, mode)
        if doc:
            return logprob, T
        else:
            return np.exp(-logprob / T)

    def __nextChar(self, context: Tuple, params: Dict, mode='add_k') -> str:
        """
        Get the next character from the context, the contect is a bi-gram if we use a tri-gram model
        Args:
            context: the ngram context to generate the next character
            params: parameters for the model to compute the probabilty
            mode: smoothing method to compute the probability, by default 'add_k'

        Returns:
            (str): next character
        """
        probs = np.zeros(self.vocab_size)
        for i in range(len(self.vocab)):
            ngram = context + (self.vocab[i],)
            if mode == 'add_k':
                probs[i] = self.prob(ngram, params['k'])
            elif mode == "inter":
                probs[i] = self.probInt(ngram, params['lambdas'])
        probs /= np.sum(probs)
        chr_idx = np.random.multinomial(1, probs).argmax()
        return self.vocab[chr_idx]

    def generateSentence(self, params: Dict, start='', mode='add_k') -> str:
        """
        Generate a sentence based on parameters/mode and the starting text.
        The starting is splited into charater if not empty and add the sentence starting marker.
        Args:
            params: parameters for the model to compute the probabilty
            start: starting character or text to generate sentence
            mode: smmothing method to compute the probability, by default 'add_k'

        Returns:
            (str): generated sentence
        """

        sentence = [self.tokenizer.start]
        if start is not None:
            sentence += list(start)
        while sentence[-1] != self.tokenizer.end:
            context = sentence[-(self.orders - 1):]
            sentence.append(self.__nextChar(tuple(context), params, mode))
        return "".join(sentence)


def tune_k(model: NGModel, val_file: str, k_list: np.ndarray) -> List[float]:
    """
    Tune the smoothing parameters of the model with Add-k smoothing.
    Args:
        model: Model to be tuned
        val_file: validation file path
        k_list: list of k

    Returns:
        (list): list of perplexities for each k considered
    """
    perplexities = []
    for k in k_list:
        sums, counts = 0, 0
        with open(val_file) as corpus:
            for line in corpus:
                p, c = model.perplexity(line, params={'k': k}, doc=True, mode='add_k')
                sums += p
                counts += c
        perplexities.append(np.exp(-sums / counts))
    return perplexities


def evalModels(models: Tuple[NGModel], val_file: str, params: Dict, mode: str = 'add_k') -> np.ndarray:
    """
    Evaluate the models with the given parameters/mode on with a validation set
    Args:
        models: list of models to be evaluated
        val_file: validation file path
        params: parameters for the model to compute the probabilty
        mode: smooting mode, by default 'add_k'

    Returns:
        (numpy.ndarray): list of perplexities on each model
    """
    sums = np.zeros(len(models))
    counts = np.zeros(len(models))
    with open(val_file) as corpus:
        for line in corpus:
            for i, model in enumerate(models):
                p, t = model.perplexity(line, params=params[model.name], doc=True, mode=mode)
                sums[i] += p
                counts[i] += t
    _h = -sums / counts
    docpp = np.exp(_h)
    return docpp


def make_cmf(models: Tuple[NGModel], val_files: Tuple[str], params: Dict, mode='add_k', title: str = '', save:Optional[str]=None) -> None:
    """
    Make a confusion matrix of all models with the given parameters/mode on a several validation set
    Args:
        models: List of models to be evaluated
        val_files: List of validation file paths to evaluate each model
        params: parameters for the model to compute the probabilty
        mode: smooting mode, by default 'add_k'
        title: Title of the confusion matrix plot

    Returns:
        None
    """
    n = len(models)
    labels = list(model.name for model in models)
    cmf = np.array([])
    for val_file in val_files:
        cmf = np.append(cmf, evalModels(models, val_file, params, mode))
    cmf = cmf.reshape(n, -1)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cmf, annot=True)
    plt.xticks(np.arange(len(labels)) + 0.5, labels)
    plt.yticks(np.arange(len(labels)) + 0.5, labels)
    plt.title(title)
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    plt.show()
