import time
import math
import itertools

import string
from pylab import *


def cur_time():
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def token_strip(string, instance_nlp):
    doc = instance_nlp(clean_text(string))
    # return [instance_lemmatizer(doc[i].string.strip(), doc[i].pos)[0] for i in range(len(doc))]
    return [doc[i].string.strip() for i in range(len(doc)) if doc[i].string.strip()]

def token_lemma(text, instance_nlp, instance_lemmatizer):
    doc = instance_nlp(clean_text(text))
    return [instance_lemmatizer(doc[i].string.strip(), doc[i].pos)[0] for i in range(len(doc)) if doc[i].string.strip()]
    # return [doc[i].string.strip() for i in range(len(doc)) if doc[i].string.strip()]


def clean_text(text):
    lower = text.lower()
    remove_punctuation_map = dict((ord(char), '') for char in string.punctuation)
    return lower.translate(remove_punctuation_map)


def draw_confusion_matrix(data: '2 dimension matrix', labels, path_name, show=False):
    norm_data = []  # values of data are normalized to 0~1
    for r in data:
        sum_r = sum(r)
        norm_r = list(map(lambda x: x / sum_r, r))
        norm_data.append(norm_r)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    res = ax.imshow(np.array(norm_data), cmap=plt.cm.gray,
                    interpolation='nearest')
    fig.colorbar(res)
    width = len(norm_data)
    height = len(norm_data[0])

    _, labels = plt.xticks(range(width), labels[:width])
    for t in labels:
        t.set_rotation(60)
    plt.yticks(range(height), labels[:height])

    if show: plt.show()
    plt.savefig(path_name, format='png')

def read_confusion_data(file_name):
    pres, exps = [], []
    with open(file_name, 'r') as f:
        for line in f:
            _, pre, exp, *_ = line.split('\t')
            pre = round(float(pre)*2)
            if pre > 10: pre = 10
            if pre < 0: pre = 0
            exp = round(float(exp)*2)
            pres.append(pre)
            exps.append(exp)
    return pres, exps

def plot_confusion_matrix(cm, classes, path_name,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # cm = cm.astype('float') / cm.sum()
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    # print(cm)
    plt.clf()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*100, fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path_name, format='png')
    # plt.show()

def calculate_F1(file_name):
    f1, found_correct, found, truth = {}, {}, {}, {}

    with open(file_name) as f:
        lines = f.readlines()
        n = len(lines)
        for line in lines:
            _, pre, exp, *_ = line.split('\t')
            pre = round(float(pre))
            if pre > 5: pre = 5
            if pre < 0: pre = 0
            exp = round(float(exp))
            found[pre] = found.get(pre, 0) + 1
            truth[exp] = truth.get(exp, 0) + 1
            if pre == exp:
                found_correct[exp] = found_correct.get(exp, 0) + 1
    for score in truth:
        if score not in found:
            precision = 0
        else:
            precision = found_correct.get(score, 0) / found[score]
        recall = found_correct.get(score, 0) / truth[score]
        if precision * recall == 0:
            f1[score] = 0
        else:
            f1[score] = 2 * precision * recall / (recall + precision)
    wf1 = 1/n * sum([truth[score] * f1.get(score, 0) for score in truth])
    return wf1

def text_weight_color(text, weights):
    html = []
    tokens = text.strip().split(' ')
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        weight = float(weights.get(token, 0))
        b, r = 0, 0
        if weight < 0:
            r = 255
        if weight > 0:
            b = 255
        html.append("<span style='border-radius:5px;background-color:rgba({},0,{},{})'>{}</span>".format(r, b, abs(weight), token))
    return '<div>' + ' '.join(html) + '</div>'