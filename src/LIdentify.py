from .utils import sort_dict

class LIdentify:
    """
        Language Identifiers
    """
    def __init__(self, args):
        self.models = [*args]

    def counts_scorring(self, unknown, order=3):
        score = {model.name: 0 for model in self.models}
        for gram in unknown:
            for model in self.models:
                score[model.name] += model.ngrams[order].get(gram, 0)
        return sort_dict(score)
