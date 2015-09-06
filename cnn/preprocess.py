# Imports
from gensim.models.word2vec import Word2Vec
import cPickle as cpkl
import pickle as pkl
import numpy as np
import nltk
import gc

# Initialize
tr_ratio = 0.8

# Paths
w2v_mod_path = './data/GoogleNews-vectors-negative300.bin'
data_path = './data/sentences.pickle'

# Load data
print "\nunpickling training data..."
file = open(data_path, 'r')
data_dict = pkl.load(file)

# Split
N = len(data_dict)
N_tr = tr_ratio*N
N_ts = N-N_tr
max_len = max([len(nltk.tokenize.word_tokenize(x)) for x in data_dict.keys()])

# Load word2vec model
print '\nloading w2v model...'
model = Word2Vec.load_word2vec_format(w2v_mod_path, binary=True)  # C binary format

# Convert to matrix format
print '\nconverting to matrix format...'
data = np.zeros((N, 300, max_len))
y = np.asarray(data_dict.values(), dtype=np.int32)

sent_count = 0
tok_count = 0
for sentence in data_dict.keys():
    for token in nltk.tokenize.word_tokenize(sentence):
        try:
            data[sent_count, :, tok_count] = model[token]
        except KeyError:
            pass
        except Exception as ex:
            template = "An exception of type {0} occured. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print message
        finally:
            tok_count += 1
    sent_count += 1
    tok_count = 0

# shuffle
# print '\nshuffling...'
# rng_state = np.random.get_state()
# np.random.shuffle(data)
# np.random.set_state(rng_state)
# np.random.shuffle(y)

# dereference word2vec model
model = -1
gc.collect()

# pickle data
print '\npickling the matrix...'
cpkl.dump((data[0:N_tr,:], y[0:N_tr]), open('./data/train_binary.pkl', "wb"), protocol=2)
cpkl.dump((data[N_tr:,:], y[N_tr:]), open('./data/test_binary.pkl', "wb"), protocol=2)