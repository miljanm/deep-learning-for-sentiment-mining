import argparse
import nltk
import simplejson as json
import sys
from pprint import pprint
from preprocessing.neural_feature_design import extract_neural_features
from models.conv_net_model import predict
from gensim.models.word2vec import Word2Vec
from itertools import islice
from collections import Counter, defaultdict
import string
import string, pdb
from nltk import RegexpTokenizer
from csv import reader


__author__ = 'miljan'


def plot(lengths):
    import numpy as np
    import matplotlib.pyplot as plt

    temp = zip(*lengths)
    keys = temp[0]
    values = temp[1]
    print sum(values[:43])
    print sum(values)
    print sum(values[:43])/float(sum(values))
    print lengths

    width = 0.35
    ind = np.arange(0, len(values)*width, width)

    fig, ax = plt.subplots()
    ax.set_xlim(right=50)

    ax.bar(ind, values, width, color='b')

    # add some text for labels, title and axes ticks
    ax.set_ylabel('# of sentences')
    ax.set_xlabel('# of sentence tokens')
    ax.set_title('Distribution of sentence lengths in the SSTb dataset')

    plt.show()


def _read_json_articles():
    text = json.load(open('../data/articles/1000-articles.json'))
    return text


def analyze_dataset():
    l_sentences = []
    with open('/Users/miljan/PycharmProjects/thesis-shared/data/pang_and_lee_data/rt-negative.txt') as file1:
        r = reader(file1, dialect='excel-tab')
        for row in r:
            l_sentences.append(row[0])
    with open('/Users/miljan/PycharmProjects/thesis-shared/data/pang_and_lee_data/rt-positive.txt') as file2:
        r = reader(file2, dialect='excel-tab')
        for row in r:
            l_sentences.append(row[0])

    # chunk the given text into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    d_lengths = defaultdict(int)
    tokenizer2 = RegexpTokenizer(r'\w+')

    # clean sentences from punctuation
    l_sentences = [''.join(ch for ch in sent if ch not in set(string.punctuation)) for sent in l_sentences]
    l_sentences = [len(tokenizer2.tokenize(sen)) for sen in l_sentences]
    total_sent = len(l_sentences)
    d_lengths = Counter(l_sentences)

    print total_sent
    lengths = sorted(d_lengths.iteritems(), key=lambda key_value: int(key_value[0]))
    plot(lengths)

def analyze_articles():
    json_document = _read_json_articles()
    l_articles = [json_document[i]['_source']['content'] for i in range(len(json_document))]

    # chunk the given text into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    d_lengths = defaultdict(int)
    tokenizer2 = RegexpTokenizer(r'\w+')
    total_sent = 0

    for article in l_articles:
        l_sentences = tokenizer.tokenize(article)
        # clean sentences from punctuation
        l_sentences = [''.join(ch for ch in sent if ch not in set(string.punctuation)) for sent in l_sentences]
        l_sentences = [len(tokenizer2.tokenize(sen)) for sen in l_sentences]
        total_sent += len(l_sentences)
        d_counts = Counter(l_sentences)
        for key in d_counts.keys():
            d_lengths[str(key)] += d_counts[key]
    print total_sent
    lengths = sorted(d_lengths.iteritems(), key=lambda key_value: int(key_value[0]))
    plot(lengths)


if __name__ == '__main__':
    analyze_articles()

