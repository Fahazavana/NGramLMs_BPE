import numpy as np
class LIdentify:
    """
        Language Identifiers
    """

    def __init__(self, models, alphas, mode, param):
        self.models = models
        self.alphas = alphas
        self.mode = mode
        self.param = param
        self.idxs = {model.name: i for i, model in enumerate(models)}

    def predict(self, unknown, order=3):
        score = np.zeros(len(self.models))
        for i, model in enumerate(self.models):
            score[i] = model.perplexity(unknown, params=self.param[model.name], mode=self.mode, doc=False)
        return np.argmin(score)


