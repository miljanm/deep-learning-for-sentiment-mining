from csv import reader
from os.path import join
import gensim
from gensim.models.word2vec import Word2Vec
import numpy as np
import nltk
import gc
import string
import bloscpack as bp
from ngram_feature_design import get_label
import pdb


def _create_feature_matrix(s_sentence, w2v_model, w2v_size, max_len):
    tok_count = 0
    np_data = np.zeros((w2v_size, max_len))
    l_sent_tokens = nltk.tokenize.word_tokenize(s_sentence)
    for token in l_sent_tokens:
        if len(l_sent_tokens) > max_len:
            # sentence has more tokens than max_len, input too big for the provided neural network
            return np_data, False
        try:
            np_data[:, tok_count] = w2v_model[token]
        except KeyError:
            pass
        except Exception as ex:
            template = "An exception of type {0} occured. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print message
        finally:
            tok_count += 1
    return np_data, True


def _tranform_dict_to_matrix_format(d_data, w2v_model, n_data, w2v_size, max_len):
    np_data_x = np.zeros((n_data, w2v_size, max_len), dtype=np.float32)
    np_data_y = np.asarray(d_data.values(), dtype=np.int32)

    sent_count = 0
    for sentence in d_data.keys():
        np_data_x[sent_count, :, :], b_success = _create_feature_matrix(sentence, w2v_model, w2v_size, max_len)
        sent_count += 1
    return np_data_x, np_data_y


def _transform_list_to_matrix_format(l_sentences, w2v_model, w2v_size, max_len):
    np_data = np.zeros((len(l_sentences), w2v_size, max_len), dtype=np.float32)
    sent_count = 0
    count_sent_too_long = 0
    for sentence in l_sentences:
        np_data_temp, b_success = _create_feature_matrix(sentence, w2v_model, w2v_size, max_len)
        # sentence was longer than max_len tokens, skip it
        if not b_success:
            count_sent_too_long += 1
            continue
        np_data[sent_count, :, :] = np_data_temp
        sent_count += 1
    if count_sent_too_long > 0:
       print "Number of sentences longer than max_len tokens is: ", count_sent_too_long
    return np_data


def _load_data_into_dict(s_data_path, l_exclude_punct, number_of_classes):
    """ Load the tsv data file into a dict with key:value pairs being sentence:sentiment

    :param s_data_path: system path to the data file to be loaded
    :param l_exclude_punct: if punctuation should be excluded or not
    :param number_of_classes: number of classes, can be 2, 3 or 5
    :return:
    """
    with open(s_data_path) as tsv:
        file_reader = reader(tsv, dialect="excel-tab")
        # skip title line
        file_reader.next()

        d_data = {}
        l_lengths = []
        for line in file_reader:
            s_sent = ''.join(ch for ch in line[2] if ch not in l_exclude_punct)
            if number_of_classes == 5:
                # get sentiment label 0-4
                label = int(line[3])
            else:
                # get sentiment label either 0 (negative), 1 (positive), 2 (neutral)
                label = get_label(int(line[3]), binary=True)
                # if just binary case, skip for neutral
                if number_of_classes == 2 and label == 2:
                    continue
            d_data[s_sent] = label
            # add length of each sentence to find the longest sentence
            l_lengths.append(len(nltk.tokenize.word_tokenize(s_sent)))
    return d_data, l_lengths


def _pickle_data_matrices(np_train_x, np_train_y, np_test_x, np_test_y,
                         s_train_x_path, s_train_y_path, s_test_x_path, s_test_y_path):
    """
    Pickles given train and test, x and y matrices, into given .

    :param np_train_x: numpy matrix, training data
    :param np_train_y: numpy matrix, training labels
    :param np_test_x:  numpy matrix, testing data
    :param np_test_y:  numpy matrix, testing labels
    :return: None
    """
    print "\npickling data matrices"
    # pickling training data
    bp.pack_ndarray_file(np_train_x, s_train_x_path)
    bp.pack_ndarray_file(np_train_y, s_train_y_path)
    # pickling testing data
    bp.pack_ndarray_file(np_test_x, s_test_x_path)
    bp.pack_ndarray_file(np_test_y, s_test_y_path)


def _load_word2vec_model(w2v_mod_path):
    return Word2Vec.load_word2vec_format(w2v_mod_path, binary=True)


def _load_glove_model(glove_path):

    def _any2unicode(text, encoding='utf8', errors='strict'):
        """Convert a string (bytestring in `encoding` or unicode), to unicode."""
        if isinstance(text, unicode):
            return text
        return unicode(text.replace('\xc2\x85', '<newline>'), encoding, errors=errors)
    #     return unicode(text, encoding, errors=errors)

    gensim.utils.to_unicode = _any2unicode
    return Word2Vec.load_word2vec_format(glove_path, binary=False)



def get_neural_processed_test_data(number_of_classes, strip_punctuation=True, is_shuffled=False, is_pickled=False, debug=False):
    """
    Creates matrices of training and testing data, created using word2vec pretrained model, of dimensionality 300.

    :param binary: boolean, choose between 5 class and 2 class dataset
    :param is_shuffled: boolean, choose if training matrix is shuffled
    :param strip_punctuation: boolean, choose if punctuation should be remove from training sentences
    :param debug: boolean, chose if verbose mode is on
    :return: training and testing, X and y numpy matrices, in the following order
        train_x, train_y, test_x, test_y
    """
    # debug info
    if debug:
        print "\nPrinting chosen options..."
        print "strip punctuation: ", strip_punctuation
        print "shuffling: ", is_shuffled

    # set paths
    train_path = "../data/kaggle_data/train.tsv"
    test_path = "../data/kaggle_data/kaggle_full_test.tsv"
    w2v_mod_path = '/Users/miljan/PycharmProjects/thesis-shared/data/GoogleNews-vectors-negative300.bin'
    glove_path = '../data/GloVe/glove.6B.300d.txt'
    w2v_size = 300
    max_len = 43

    # list of punctuation to exclude
    l_exclude_punct = set(string.punctuation) if strip_punctuation else []

    # load train data
    print "\nLoading train data for neural features..."
    d_train, l_train_lengths = _load_data_into_dict(train_path, l_exclude_punct, number_of_classes)

    # load test data
    print "\nLoading test data for neural features..."
    d_test, l_test_lengths = _load_data_into_dict(test_path, l_exclude_punct, number_of_classes)

    # Setting dimensions
    n_train = len(d_train)
    n_test = len(d_test)
    # max_len = max(max(l_train_lengths), max(l_test_lengths))

    # debug info
    if debug:
        print "\nPrinting debug info..."
        print "training samples: ", n_train
        print "testing samples: ", n_test
        print "longest train sentences lengths: ", sorted(l_train_lengths)[-20:]
        print "longest test sentences lengths: ", sorted(l_test_lengths)[-20:]
        print "max sentence length: ", max_len

    # load neural model
    print '\nloading w2v model...'
    w2v_model = _load_word2vec_model(w2v_mod_path) # _load_glove_model(glove_path)

    # convert training data to matrix format
    print '\nconverting training data to matrix format...'
    np_train_x, np_train_y = _tranform_dict_to_matrix_format(d_train, w2v_model, n_train, w2v_size, max_len)
    # dereference training data structures
    d_train = -1
    gc.collect()

    # convert testing data to matrix format
    print '\nconverting testing data to matrix format...'
    np_test_x, np_test_y = _tranform_dict_to_matrix_format(d_test, w2v_model, n_test, w2v_size, max_len)
    # dereference testing data structures
    d_test = -1
    gc.collect()

    if debug:
        print '\nfirst 50 training sentences have labels:'
        print np_train_y[:50]
        print '\nfirst 50 testing sentences have labels:'
        print np_test_y[:50]

    # shuffling data
    if is_shuffled:
        print '\nShuffling...'
        rng_state = np.random.get_state()
        np.random.shuffle(np_train_x)
        np.random.set_state(rng_state)
        np.random.shuffle(np_train_y)

    # save the data files
    if is_pickled:
        _pickle_data_matrices(np_train_x, np_train_y, np_test_x, np_test_y,
                             '../data/blp/train_x.blp', '../data/blp/train_y.blp',
                             '../data/blp/test_x.blp', '../data/blp/test_y.blp')

    return np_train_x, np_train_y, np_test_x, np_test_y


def extract_neural_features(s_text, w2v_model, strip_punctuation=True):
    """
    Function which operates on a single text and returns a matrix with neural features extracted.

    :param s_text: (string), text to be processed
    :param strip_punctuation (boolean), whether or not to strip punctuation
    :return: 3d numpy matrix of shape (no_of_sentences, vector_size, max_sent_length) with neural features
    """
    # chunk the given text into sentences
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    l_sentences = tokenizer.tokenize(s_text)

    # list of punctuation to exclude
    l_exclude_punct = set(string.punctuation) if strip_punctuation else []
    # clean sentences from punctuation
    l_sentences = [''.join(ch for ch in sent if ch not in l_exclude_punct) for sent in l_sentences]

    # word2vec vector dimension
    w2v_size = 300
    # maximum size of the sentence (max no of tokens in sentence allowed)
    max_len = 43

    # convert each sentence to a feature matrix
    np_data = _transform_list_to_matrix_format(l_sentences, w2v_model, w2v_size, max_len)

    return np_data


if __name__ == '__main__':
    # get_neural_processed_test_data(2, debug=True, is_pickled=True)
    pass