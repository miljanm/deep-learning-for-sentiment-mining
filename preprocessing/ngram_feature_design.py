from nltk import RegexpTokenizer
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from csv import reader
from collections import defaultdict
import operator
import numpy as np


__author__ = 'miljan'


def _get_ngram_features(infile, ngram_size):
    """
    Returns a dictionary containing ngrams and counts observed in a given file

    :param infile: file to be analysed
    :param ngram_size: ngram size
    :return: dict of ngrams/counts
    """
    # tokenizer which remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')
    # dictionary on ngrams and counts
    d_ngrams = defaultdict(int)
    # stopwords
    stops = set(stopwords.words("english"))
    # lemmatizer for stemming
    lemmatizer = WordNetLemmatizer()

    # load train data
    with open(infile) as tsv:
        file_reader = reader(tsv, dialect="excel-tab")
        # skip title line
        file_reader.next()
        for line in file_reader:
            s_text = line[2]
            # remove punctuation and tokenize
            l_text = tokenizer.tokenize(s_text)
            # remove stopwords and stem
            l_text = [lemmatizer.lemmatize(word) for word in l_text if word not in stops]
            # get the ngrams for the given line
            l_temp = ngrams(l_text, ngram_size)
            for ngram in l_temp:
                d_ngrams[ngram] += 1

    return d_ngrams


def get_label(label, binary=True):
    if binary:
        if label == 2:
            return label
        else:
            return 0 if label < 2 else 1
    else:
        return label


def get_ngram_processed_data(data_path, ngram_size=2, top_ngrams=None, binary=False):
    """
    Processes a given tsv file and return a numpy feature matrix and numpy label list

    :param data_path: file to be processed
    :param ngram_size: ngram size
    :param top_ngrams: number of most frequent bigram to look for
    :param binary: binary classification (only look at example with positive and negative sentiment)
    :return: numpy
    """
    # dictionary of ngrams with their frequencies from the training set
    d_all_ngrams = _get_ngram_features('../data/kaggle_data/train.tsv', ngram_size)
    # get only the top n ngrams
    if top_ngrams:
        # sort by frequency
        sorted_ngrams = sorted(d_all_ngrams.items(), key=operator.itemgetter(1), reverse=True)
        # get the top n most frequent ngrams
        l_top_n_ngrams = [x[0] for x in sorted_ngrams[:top_ngrams]]
        # map each ngram to a unique integer
        d_all_ngrams = dict(zip(l_top_n_ngrams, range(0, len(l_top_n_ngrams))))
        if top_ngrams > len(d_all_ngrams):
            raise ValueError('Number of top ngrams is greater than number of all found ngrams.')
    # get all ngrams
    else:
        # map each ngram to a unique integer
        d_all_ngrams = dict(zip(d_all_ngrams.keys(), range(0, len(d_all_ngrams))))

    # data and label lists
    l_x = []
    l_y = []

    # tokenizer which remove punctuation
    tokenizer = RegexpTokenizer(r'\w+')

    # load train data
    with open(data_path) as tsv:
        file_reader = reader(tsv, dialect="excel-tab")
        # skip title line
        file_reader.next()

        for line in file_reader:
            # get the label for the given line
            label = get_label(int(line[3]), binary=binary)
            # if we are doing binary classification, ignore neutral samples
            if binary and label == 2:
                continue
            l_y.append(label)
            # get the ngrams for the given line
            l_temp = ngrams(tokenizer.tokenize(line[2]), ngram_size)
            # get the indices of all detected ngrams
            l_detected_ngrams = []
            for entry in l_temp:
                try:
                    l_detected_ngrams.append(d_all_ngrams[entry])
                except KeyError:
                    pass
            l_x.append(l_detected_ngrams)
    # convert training matrix into numpy matrix
    np_x = np.zeros((len(l_x), len(d_all_ngrams)), dtype=np.int32)
    for i, row in enumerate(l_x):
        np_x[i, row] = 1
    np_y = np.asarray(l_y, dtype=np.int32)

    return np_x, np_y


if __name__ == '__main__':
    pass
