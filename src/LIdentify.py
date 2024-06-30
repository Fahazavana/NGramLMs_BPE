from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .NgramModel import NGModel


class LIdentify:
    """
        Language Identifiers
        - Language prediction for a given sentece
    """

    def __init__(self, models: Tuple[NGModel], mode: str, params: Dict) -> None:
        self.models = models
        self.mode = mode
        self.param = params
        self.idxs = {model.name: i for i, model in enumerate(models)}

    def predict(self, unknown: str, order=3) -> int:
        score = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            score[i] = model.perplexity(unknown, params=self.param[model.name], mode=self.mode, doc=False)
        return np.argmin(score)


def cmf_acc_matrix(identifiers: LIdentify, x: List[str], y: List[str]) -> np.ndarray:
    """
    Calculates the confusion matrix for the test data, using Language identinfier
    Args:
        identifiers: Language identifier class
        x: test data
        y: test data label

    Returns:
        cmx: confusion matrix
    """
    n = len(set(y))
    cmx = np.zeros((n, n))
    for xi, yi in zip(x, y):
        yihat = identifiers.predict(xi)
        cmx[yihat, identifiers.idxs[yi]] += 1
    return cmx


def plot_acc_cmf(cmf: np.ndarray, acc: float, labels: List[str], title: str = '',
             figsize: Tuple[int, int] = (6, 5), fmt='.0f') -> None:
    """"
        Plots the confusion matrix  of the accurracy for the test data, using Language identinfier
    """
    plt.figure(figsize=figsize)
    sns.heatmap(cmf, annot=True, fmt=fmt)
    plt.xticks(np.arange(len(labels)) + 0.5, labels)
    plt.yticks(np.arange(len(labels)) + 0.5, labels)
    plt.title(fr"Confusion matrix - {title} - acc = {acc:.2f}%")
    plt.xlabel("Predicted")
    plt.ylabel("True")
