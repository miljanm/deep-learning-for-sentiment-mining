#!/usr/bin/env python

"""
Convolutionary neural network for language, trained on sentiment mining problem
"""
import logging
import argparse
import datetime
import traceback
import numpy as np
import gc, sys, pdb
import bloscpack as bp
from copy import deepcopy
import chainer.functions as F
import six.moves.cPickle as pickle
from chainer import cuda, Variable, FunctionSet, optimizers
from sklearn.metrics import confusion_matrix

import parameter_search2
from parameter_search2 import gen_point


def evaluate_results(x_test, y_test, N_test, batchsize, max_len, print_conf_matrix=False):
    '''
    Evaluate model on test set.
    :param x_test:
    :param y_test:
    :param N_test:
    :param batchsize:
    :param max_len:
    :return:
    '''

    # reshape data to match chainer format
    x_test = np.reshape(x_test, (x_test.shape[0], 1, word_vector_size, max_len))

    # evaluation
    sum_accuracy = 0
    sum_loss     = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]
        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        loss, acc = forward(x_batch, y_batch, print_conf_matrix=False)

        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    # get conf matrix for the whole test set
    if args.gpu >= 0:
        x_batch = cuda.to_gpu(x_test)
        y_batch = cuda.to_gpu(y_test)
    loss, acc = forward(x_batch, y_batch, print_conf_matrix=print_conf_matrix)

    print 'test mean loss={}, accuracy={}'.format(
        sum_loss / N_test, sum_accuracy / N_test)
    return sum_accuracy / N_test


def forward(x_data, y_data, print_conf_matrix=False):
    '''
    Neural net architecture
    :param x_data:
    :param y_data:
    :param train:
    :return:
    '''
    x, t = Variable(x_data), Variable(y_data)

    h1 = F.relu(model.l1(x))
    h1 = F.max_pooling_2d(h1,max_pool_window_1,stride=max_pool_stride_1)

    h2 = F.dropout(F.relu(model.l2(h1)))
    h2 = F.average_pooling_2d(h2, avg_pool_window_2, stride=avg_pool_stride_2)
    h2 = F.max_pooling_2d(h2,max_pool_window_2,stride=max_pool_stride_2)

    y = model.l3(h2)

    # display confusion matrix
    if print_conf_matrix:
        print confusion_matrix(cuda.to_cpu(t.data), cuda.to_cpu(y.data).argmax(axis=1))

    return F.softmax_cross_entropy(y, t), F.accuracy(y, t)


def _compute_dimensions(x_data, y_data):
    '''
    Compute network dimension
    :param x_data:
    :param y_data:
    :return:
    '''

    x, t = Variable(x_data), Variable(y_data)

    h1 = F.relu(model.l1(x))
    h1 = F.max_pooling_2d(h1,max_pool_window_1,stride=max_pool_stride_1)

    h2 = F.dropout(F.relu(model.l2(h1)))
    h2 = F.average_pooling_2d(h2, avg_pool_window_2, stride=avg_pool_stride_2)
    h2 = F.max_pooling_2d(h2,max_pool_window_2,stride=max_pool_stride_2)

    return h2.data.shape


def _load_train_data(divisor, offset):
    '''
    Load subset of data
    :param divisor:
    :param offset:
    :return:
    '''

    start = offset / float(divisor)
    end = (1 + offset) / float(divisor)

    x_train = bp.unpack_ndarray_file(data_path_train_x)
    data_temp = np.empty_like(x_train[np.floor(x_train.shape[0]*start):np.floor(x_train.shape[0]*end), :, :])
    np.copyto(data_temp, x_train[np.floor(x_train.shape[0]*start):np.floor(x_train.shape[0]*end), :, :])
    x_train = -1
    gc.collect()
    x_train = data_temp
    y_train = bp.unpack_ndarray_file(data_path_train_y)
    y_train = deepcopy(y_train[np.floor(y_train.shape[0]*start):np.floor(y_train.shape[0]*end)])

    return x_train, y_train


# Parsing
parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

# setup logging
logging.basicConfig(filename='logs/hyperp_fine_glove_200x32.log', filemode='a+', level=logging.DEBUG)
# logging.disable(logging.CRITICAL)

# Paths
data_path_train_x = './data/blp/train_x.blp'
data_path_train_y = './data/blp/train_y.blp'
data_path_test_x = './data/blp/test_x.blp'
data_path_test_y = './data/blp/test_y.blp'

root = '../thesis-shared/data/blp/glove/200d/five_classes/'
# Paths
data_path_train_x = root + 'train_x.blp'
data_path_train_y = root + 'train_y.blp'
data_path_test_x = root + 'test_x.blp'
data_path_test_y = root + 'test_y.blp'

# Hyperparameters
batchsize = 50
n_epoch   = 5
data_ratio = 1.3
b_conf_matrix = True
word_vector_size = 200
sentence_length = 32

# Load train data
print "\nunpickling training data..."
x_train, y_train = _load_train_data(data_ratio, 0.0)
# pdb.set_trace()

print "\nTraining matrix shape:"
print x_train.shape
print "\nTraining matrix size in bytes: "
print x_train.nbytes
# Load test data
print "\nunpickling testing data..."
x_test = bp.unpack_ndarray_file(data_path_test_x)
y_test = bp.unpack_ndarray_file(data_path_test_y)
N_test = x_test.shape[0]

print N_test

# data_info
print "\nreading info..."
N = x_train.shape[0]
dim = x_train.shape[1]
max_len = x_train.shape[2]

# reshape data to match chainer format
x_train = np.reshape(x_train, (x_train.shape[0], 1, word_vector_size, max_len))

# Hyper search params
variances = [3.3, 5.0, 1.5, 1.0, 1.5, 1.0]
l_best_params = [15, 12, 1, 7, 1, 5]
best_accuracy = 0
pool_size = 20
epsilon = 0.05

for attempt in range(1):
    print '\n\n Current attempt is: ', attempt

    ### optimization
    l_current_parameters = map(int, gen_point(l_best_params, variances, pool_size, epsilon))
    # best so far
    l_current_parameters = [15, 21, 2, 6, 3, 7]
    ch_out_1, ch_out_2, \
    filter_1_x, filter_1_y, \
    filter_2_x, filter_2_y = l_current_parameters
    print 'Parameters chosen: ', l_current_parameters

    try:
        # Compute network dimensions
        print "\nplanning dimension of the neural network..."
        h_1 = dim
        w_1 = max_len
        ch_in_1 = 1
        # ch_out_1 = 15
        filter_1 = (filter_1_x, filter_1_y)
        stride_1 = (1,1)
        padding_1 = (0,6)

        max_pool_window_1 = (1,11)
        max_pool_stride_1 = (1,5)

        h_2_beforemax = ((h_1+2*padding_1[0]-filter_1[0])/stride_1[0]+1)
        h_2 = ((h_1+2*padding_1[0]-filter_1[0])/stride_1[0]+1)/max_pool_stride_1[0]
        w_2_beforemax = ((w_1+2*padding_1[1]-filter_1[1])/stride_1[1]+1)
        w_2 = ((w_1+2*padding_1[1]-filter_1[1])/stride_1[1]+1)/max_pool_stride_1[1]

        ch_in_2 = ch_out_1
        # ch_out_2 = 12
        filter_2 = (filter_2_x, filter_2_y)
        stride_2 = (1,1)
        padding_2 = (0,4)

        avg_pool_window_2 = (2,1)
        avg_pool_stride_2 = (2,1)

        max_pool_window_2 = (1,7)
        max_pool_stride_2 = (1,4)

        h_3_beforeavg = ((h_2+2*padding_2[0]-filter_2[0])/stride_2[0]+1)
        h_3_beforemax = (((h_2+2*padding_2[0]-filter_2[0])/stride_2[0]+1)/avg_pool_stride_2[0])
        h_3 = (((h_2+2*padding_2[0]-filter_2[0])/stride_2[0]+1)/avg_pool_stride_2[0])/max_pool_stride_2[0]
        w_3_beforeavg = ((w_2+2*padding_2[1]-filter_2[1])/stride_2[1]+1)
        w_3_beforemax = (((w_2+2*padding_2[1]-filter_2[1])/stride_2[1]+1)/avg_pool_stride_2[1])
        w_3 = (((w_2+2*padding_2[1]-filter_2[1])/stride_2[1])/avg_pool_stride_2[1])/max_pool_stride_2[1]

        linear_in_3 = ch_out_2*h_3*w_3
        linear_out_3 = len(np.unique(y_train))

        # compute dimensions of the network
        model = FunctionSet(l1=F.Convolution2D(ch_in_1, ch_out_1, filter_1, stride=stride_1, pad=padding_1),
                            l2=F.Convolution2D(ch_out_1, ch_out_2, filter_2, stride=stride_2, pad=padding_2))
        t_shape = _compute_dimensions(np.zeros((100, 1, word_vector_size, sentence_length)), np.zeros((100,)))


        # Define elements of the deep convolutionary network
        model = FunctionSet(l1=F.Convolution2D(ch_in_1, ch_out_1, filter_1, stride=stride_1, pad=padding_1),
                            l2=F.Convolution2D(ch_out_1, ch_out_2, filter_2, stride=stride_2, pad=padding_2),
                            l3=F.Linear(t_shape[3]*t_shape[2]*t_shape[1], linear_out_3))

        # Setup GPU if required
        if args.gpu >= 0:
            print "\ninitializing graphical processing unit..."
            cuda.init(args.gpu)
            model.to_gpu()

        # Setup optimizer
        optimizer = optimizers.Adam()
        optimizer.setup(model.collect_parameters())

        # use momentum SGD first
        # is_sgd = False

        max_accuracy = 0
        max_accuracy_epoch = 0
        is_first_part = True
        divisor = data_ratio
        offset = 0.0
        # Learning loop
        print "\nstart training..."
        for epoch in xrange(0, n_epoch):
            print '\nepoch', epoch + 1

            # dynamic switching of optimizers
            # if epoch % 10 == 0:
            #     if is_sgd:
            #         optimizer = optimizers.AdaGrad()
            #         optimizer.setup(model.collect_parameters())
            #         is_sgd = False
            #     else:
            #         optimizer = optimizers.Adam()
            #         optimizer.setup(model.collect_parameters())
            #         is_sgd = True

            # switch between first and second part of the data
            if epoch % 100 == 0 and epoch != 0:
                print "Loading new part of the training data set..."
                # load the second part of the data
                x_train, y_train = _load_train_data(divisor, offset)
                # mod of offset depends on how many overlapping sectors of data we want
                offset = (offset + 1) % 3
                # reshape data to match chainer format
                x_train = np.reshape(x_train, (x_train.shape[0], 1, word_vector_size, max_len))
                N = x_train.shape[0]

            # Training
            perm = np.random.permutation(N)
            sum_accuracy = 0
            sum_loss = 0
            for i in xrange(0, N, batchsize):
                x_batch = x_train[perm[i:i+batchsize]]
                y_batch = y_train[perm[i:i+batchsize]]
                if args.gpu >= 0:
                    x_batch = cuda.to_gpu(x_batch)
                    y_batch = cuda.to_gpu(y_batch)

                optimizer.zero_grads()
                loss, acc = forward(x_batch, y_batch)
                loss.backward()
                optimizer.update()

                sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
                sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

            print 'train mean loss={}, accuracy={}'.format(
                sum_loss / N, sum_accuracy / N)

            eval_accuracy = evaluate_results(x_test, y_test, N_test, batchsize, max_len, print_conf_matrix=b_conf_matrix)
            if eval_accuracy > max_accuracy:
                max_accuracy = eval_accuracy
                max_accuracy_epoch = epoch + 1

        print '\n Max accuracy was {} in epoch {}'.format(max_accuracy, max_accuracy_epoch)
        log_string = str(l_current_parameters) + ', ' + str(max_accuracy)
        logging.info(log_string)
        parameter_search2.history_x.append(l_current_parameters)
        parameter_search2.history_y.append(max_accuracy)
        parameter_search2.clf.fit(parameter_search2.history_x, parameter_search2.history_y)
        if max_accuracy > best_accuracy:
            best_accuracy = max_accuracy
            l_best_params = l_current_parameters

    except KeyboardInterrupt:
        evaluate_results(x_test, y_test, N_test, batchsize, max_len)
        print '\n Max accuracy was {} in epoch {}'.format(max_accuracy, max_accuracy_epoch)
        sys.exit(0)

    except Exception as e:
        print '\n'
        traceback.print_exc()
        print e
        print '\n'
        continue

    finally:
        model.to_cpu()
        pickle.dump((model,
                     max_pool_window_1, max_pool_stride_1,
                     avg_pool_window_2, avg_pool_stride_2,
                     max_pool_window_2, max_pool_stride_2),
                    open('./models/' + str(datetime.datetime.now()), 'wb'))

