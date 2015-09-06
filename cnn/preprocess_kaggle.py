from csv import reader
from gensim.models.word2vec import Word2Vec
import numpy as np
import nltk
import gc, pdb
import string
import bloscpack as bp

# options
print "\nPrinting chosen options..."
strip_punct = True
w2v_size = 300
shuffling = False
print "strip punctuation: ", strip_punct
print "word2vec size: ", w2v_size
print "shuffling: ", shuffling

# paths
print "\nInitialize paths and data structures..."
train_path = "./data/kaggle_data/train.tsv"
test_path = "./data/kaggle_data/kaggle_full_test.txt"
w2v_mod_path = './data/GoogleNews-vectors-negative300.bin'

# data structures
train_dict = {}
test_dict = {}
train_lengths = []
test_lengths = []
exclude = set(string.punctuation) if strip_punct else []

# load train data
print "\nLoading train data..."
with open(train_path) as tsv:

    r = reader(tsv, dialect="excel-tab")
    r.next()
    for line in r:
        s = ''.join(ch for ch in line[2] if ch not in exclude)
        train_dict[s]=line[3]
        train_lengths.append(len(nltk.tokenize.word_tokenize(s)))

# load test data
print "\nLoading test data..."
with open(test_path) as tsv:

    r = reader(tsv, dialect="excel-tab")
    r.next()
    for line in r:
        s = ''.join(ch for ch in line[2] if ch not in exclude)
        test_dict[s]=line[3]
        test_lengths.append(len(nltk.tokenize.word_tokenize(s)))

# Setting dimensions
print "\nSetting matrix dimensions..."
N_tr = len(train_dict)
N_ts = len(test_dict)
max_len = max(max(train_lengths), max(test_lengths))

# Debug info
print "\nPrinting debug info..."
print "training samples: ", N_tr
print "testing samples: ", N_ts
print "longest train sentences lengths: ", sorted(train_lengths)[-20:]
print "longest test sentences lengths: ", sorted(test_lengths)[-20:]
print "max sentence length: ", max_len

# Load word2vec model
print '\nloading w2v model...'
model = Word2Vec.load_word2vec_format(w2v_mod_path, binary=True)  # C binary format

# Convert training data to matrix format
print '\nconverting training data to matrix format...'
train_data = np.zeros((N_tr, w2v_size, max_len), dtype=np.float32)
y_train = np.asarray(train_dict.values(), dtype=np.int32)

sent_count = 0
tok_count = 0
skipped = 0
for sentence in train_dict.keys():
    for token in nltk.tokenize.word_tokenize(sentence):
        try:
            train_data[sent_count, :, tok_count] = model[token]
        except KeyError:
            skipped += 1
            pass
        except Exception as ex:
            template = "An exception of type {0} occured. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print message
        finally:
            tok_count += 1
    sent_count += 1
    tok_count = 0

print sent_count, skipped

# dereferencing training data structures
print "\nDereferencing useless training data..."
train_dict = -1
gc.collect()

# Convert testing data to matrix format
print '\nconverting testing data to matrix format...'
test_data = np.zeros((N_ts, w2v_size, max_len), dtype=np.float32)
y_test = np.asarray(test_dict.values(), dtype=np.int32)

sent_count = 0
tok_count = 0
skipped = 0
for sentence in test_dict.keys():
    for token in nltk.tokenize.word_tokenize(sentence):
        try:
            test_data[sent_count, :, tok_count] = model[token]
        except KeyError:
            skipped += 1
            pass
        except Exception as ex:
            template = "An exception of type {0} occured. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print message
        finally:
            tok_count += 1
    sent_count += 1
    tok_count = 0

print sent_count, skipped

# dereferencing useless testing data structures
print "\nDereferencing useless testing data..."
test_dict = -1
gc.collect()

# shuffling data
if shuffling:
    print '\nShuffling...'
    rng_state = np.random.get_state()
    np.random.shuffle(train_data)
    np.random.set_state(rng_state)
    np.random.shuffle(y_train)

print np.sum(train_data[:10, :, :], axis=(1, 2))
print y_train[:10]

pdb.set_trace()

# pickling training data
bp.pack_ndarray_file(train_data, './data/blp/train_x.blp')
bp.pack_ndarray_file(y_train, './data/blp/train_y.blp')

# dereferencing all training data structures
print "\nDereferencing all training data..."
train_data = -1
gc.collect()

# pickling testing data
bp.pack_ndarray_file(test_data, './data/blp/test_x.blp')
bp.pack_ndarray_file(y_test, './data/blp/test_y.blp')
