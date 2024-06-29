import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def sort_dict(dictionnary, key="val"):
    if key == "key":
        return dict(
            sorted(dictionnary.items(), key=lambda items: items[0], reverse=True)
        )
    else:
        return dict(
            sorted(dictionnary.items(), key=lambda items: items[1], reverse=True)
        )

def read_test(file_name):
    y, x = [], []
    with open(file_name) as corpus:
        for line in corpus:
            y.append(line[:2].strip())
            x.append(line[2:].strip())
    return x, y


def conf_matrix(identifiers, x, y):
    n = len(set(y))
    cmx = np.zeros((n, n))
    for xi, yi in zip(x, y):
        yihat = identifiers.predict(xi)
        cmx[yihat, identifiers.idxs[yi]] += 1
    return cmx

def tune_alpha(model, val_file, alphas):
    perplexities = []
    for alpha in alphas:
        sums, counts = 0, 0
        with open(val_file) as corpus:
            for line in corpus:
                p, c = model.perplexity(line, params={'alpha':alpha}, doc=True, mode='default')
                sums += p
                counts += c
        perplexities.append(np.exp(-sums / counts))
    return perplexities

def evalModels(models, files, params, mode='default'):
    sums = np.zeros(len(models))
    counts = np.zeros(len(models))
    with open(files) as corpus:
        for line in corpus:
            for i, model in enumerate(models):
                p, t = model.perplexity(line, params=params[model.name], doc=True, mode=mode)
                sums[i] += p
                counts[i] += t
    _h = -sums / counts
    docpp = np.exp(_h)
    return docpp

def make_cmf(models, val_files, params, mode='default', title=''):
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
    plt.show()


