class LIdentify:
    def __init__(self, args):
        self.models = [*args]

    def scoring(self, unknown, order=3):
        score = {model.name: 0 for model in self.models}
        for gram in unknown.keys():
            for model in self.models:
                score[model.name] += model.ngrams[order].get(gram, 0)
        return score
