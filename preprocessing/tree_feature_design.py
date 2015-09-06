import re
import gensim
from gensim.models.word2vec import Word2Vec
import numpy as np
import cPickle as pickle
import pdb


__author__ = 'miljan'


class SexpParser(object):

    def __init__(self, line):
        self.tokens = re.findall(r'\(|\)|[^\(\) ]+', line)
        self.pos = 0

    def parse(self):
        assert self.pos < len(self.tokens)
        token = self.tokens[self.pos]
        assert token != ')'
        self.pos += 1

        if token == '(':
            children = []
            while True:
                assert self.pos < len(self.tokens)
                if self.tokens[self.pos] == ')':
                    self.pos += 1
                    break
                else:
                    children.append(self.parse())
            return children
        else:
            return token


def convert_tree(neural_model, neural_model_size, exp):
    assert isinstance(exp, list) and (len(exp) == 2 or len(exp) == 3)

    if len(exp) == 2:
        label, leaf = exp
        try:
            word_vector = neural_model[leaf]
        except KeyError:
            word_vector = np.zeros((1, neural_model_size), dtype=np.float32)
            pass
        except Exception as ex:
            template = "An exception of type {0} occured. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print message
        return {'label': int(label), 'node': word_vector}
    elif len(exp) == 3:
        label, left, right = exp
        node = (convert_tree(neural_model, neural_model_size, left),
                convert_tree(neural_model, neural_model_size, right))
        return {'label': int(label), 'node': node}


def read_corpus(path, neural_model, neural_model_size):
    with open(path) as f:
        trees = []
        for line in f:
            line = line.strip()
            tree = SexpParser(line).parse()
            trees.append(convert_tree(neural_model, neural_model_size, tree))

        return trees


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


def _pickle_tree_sets(train, test, dev, train_path, test_path, dev_path):
    print "\npickling tree lists"
    pickle.dump(train, open(train_path, 'wb'), protocol=-1)
    pickle.dump(test, open(test_path, 'wb'), protocol=-1)
    pickle.dump(dev, open(dev_path, 'wb'), protocol=-1)


def get_neural_tree_dataset(number_of_classes, neural_model_size, is_pickled=False):
    w2v_mod_path = '/Users/miljan/PycharmProjects/thesis-shared/data/GoogleNews-vectors-negative300.bin'
    glove_path = '../data/GloVe/glove.6B.' + str(neural_model_size) + 'd.txt'

    # load neural model
    print '\nloading neural language model...'
    neural_model =_load_glove_model(glove_path)

    print '\ncreating training trees...'
    train_trees = read_corpus('../data/trees/train.txt', neural_model, neural_model_size)
    print 'creating test trees...'
    test_trees = read_corpus('../data/trees/test.txt', neural_model, neural_model_size)
    print 'creating development trees...'
    develop_trees = read_corpus('../data/trees/dev.txt', neural_model, neural_model_size)

    # save the data files
    if is_pickled:
        _pickle_tree_sets(train_trees, test_trees, develop_trees,
                          '../data/processed_trees/train.blp',
                          '../data/processed_trees/test.blp',
                          '../data/processed_trees/dev.blp')

    return train_trees, test_trees, develop_trees


def extract_neural_tree_features(neural_model, neural_model_size, sentences):
    trees = []
    for line in sentences:
        tree = SexpParser(line).parse()
        trees.append(convert_tree(neural_model, neural_model_size, tree))

    return trees



if __name__ == '__main__':
    pass

