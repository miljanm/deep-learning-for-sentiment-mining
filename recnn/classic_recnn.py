import argparse
import collections
import random
import sys
import time
import cPickle as pickle
import traceback
import numpy as np
import datetime

import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers


def traverse(node, train=True, evaluate=None, root=True):
    if isinstance(node['node'], np.ndarray):
        # leaf node
        word = np.reshape(node['node'], (1, 300))
        if args.gpu >= 0:
            word = cuda.to_gpu(word)
        loss = 0
        v = chainer.Variable(word)
    else:
        # internal node
        left_node, right_node = node['node']
        left_loss, left = traverse(
            left_node, train=train, evaluate=evaluate, root=False)
        right_loss, right = traverse(
            right_node, train=train, evaluate=evaluate, root=False)
        v = F.tanh(model.l(F.concat((left, right))))
        loss = left_loss + right_loss

    y = model.w(v)

    if train:
        label = np.array([node['label']], np.int32)
        if args.gpu >= 0:
            label = cuda.to_gpu(label)
        t = chainer.Variable(label, volatile=not train)
        loss += F.softmax_cross_entropy(y, t)

    if evaluate is not None:
        predict = cuda.to_cpu(y.data).argmax(1)
        if predict[0] == node['label']:
            evaluate['correct_node'] += 1
        evaluate['total_node'] += 1

        if root:
            if predict[0] == node['label']:
                evaluate['correct_root'] += 1
            evaluate['total_root'] += 1

    return loss, v


def evaluate(test_trees):
    result = collections.defaultdict(lambda: 0)
    for tree in test_trees:
        traverse(tree, train=False, evaluate=result)

    acc_node = 100.0 * result['correct_node'] / result['total_node']
    acc_root = 100.0 * result['correct_root'] / result['total_root']
    print(' Node accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_node, result['correct_node'], result['total_node']))
    print(' Root accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_root, result['correct_root'], result['total_root']))


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

n_epoch = 400       # number of epochs
n_units = 300       # number of units per layer
batchsize = 25      # minibatch size
n_label = 5         # number of labels
epoch_per_eval = 5  # number of epochs per evaluation

train_trees = pickle.load(open('../data/processed_trees/' + str(n_units) + 'd/train.blp', 'rb'))
test_trees = pickle.load(open('../data/processed_trees/' + str(n_units) + 'd/test.blp', 'rb'))
develop_trees = pickle.load(open('../data/processed_trees/' + str(n_units) + 'd/dev.blp', 'rb'))

model = chainer.FunctionSet(
    l=F.Linear(n_units * 2, n_units),
    w=F.Linear(n_units, n_label),
)

if args.gpu >= 0:
    cuda.init()
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.AdaGrad(lr=0.1)
optimizer.setup(model.collect_parameters())


try:
    accum_loss = 0
    count = 0
    start_at = time.time()
    cur_at = start_at
    for epoch in range(n_epoch):
        print('Epoch: {0:d}'.format(epoch))
        total_loss = 0
        cur_at = time.time()
        random.shuffle(train_trees)
        for tree in train_trees:
            loss, v = traverse(tree, train=True)
            accum_loss += loss
            count += 1

            if count >= batchsize:
                optimizer.zero_grads()
                accum_loss.backward()
                optimizer.weight_decay(0.0001)
                optimizer.update()
                total_loss += float(cuda.to_cpu(accum_loss.data))

                accum_loss = 0
                count = 0

        print('loss: {:.2f}'.format(total_loss))

        now = time.time()
        throuput = float(len(train_trees)) / (now - cur_at)
        print('{:.2f} iters/sec, {:.2f} sec'.format(throuput, now - cur_at))
        print()

        if (epoch + 1) % epoch_per_eval == 0:
            print('Train data evaluation:')
            evaluate(train_trees)
            print('Develop data evaluation:')
            evaluate(develop_trees)
            print('')

except KeyboardInterrupt:
    pass

except Exception as e:
    print '\n'
    traceback.print_exc()
    print e
    print '\n'

finally:
    print('Test evaluateion')
    evaluate(test_trees)
    model.to_cpu()
    pickle.dump(model, open('./models/' + 'RNN_' + str(n_label) +
                            'label_' + str(n_units) + '_' + str(datetime.datetime.now()), 'wb'))
    sys.exit(0)