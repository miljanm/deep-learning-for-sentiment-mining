import cPickle as pickle
import chainer
from chainer import cuda
import chainer.functions as F
import numpy as np


__author__ = 'miljan'


def predict(node, neural_model_size, root=True):
    if isinstance(node['node'], np.ndarray):
        # leaf node
        word = np.reshape(node['node'], (1, neural_model_size))
        v = chainer.Variable(word)
    else:
        # internal node
        left_node, right_node = node['node']
        left = predict(left_node, neural_model_size, root=False)
        right = predict(right_node, neural_model_size,  root=False)
        intermediate = F.tanh(model.h(F.concat((left, right))))
        v = F.tanh(model.l(F.concat((left, right))))

    y = model.w(v)

    # evaluate root label
    if root:
        predicted = cuda.to_cpu(y.data).argmax(1)
        try:
            label = node['label']
            return predicted[0], label
        except:
            pass
        return predicted[0]

    return v


# always load the recursive neural network and its parameters
model = pickle.load(open('/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/models/saved_models/RNN_5label_300', 'rb'))


if __name__ == '__main__':
    pass