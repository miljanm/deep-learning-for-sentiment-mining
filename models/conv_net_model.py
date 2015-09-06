import numpy as np
from chainer import Variable
import chainer.functions as F
import pdb
import six.moves.cPickle as pickle
from collections import Counter


# should be rewritten as part of a new Model class
def predict(np_data, model_data):
    model, \
        max_pool_window_1, max_pool_stride_1, \
        avg_pool_window_2, avg_pool_stride_2, \
        max_pool_window_2, max_pool_stride_2 = model_data
    np_data = np.reshape(np_data, (np_data.shape[0], 1, 300, 43))
    x = Variable(np_data)

    h1 = F.relu(model.l1(x))
    h1 = F.max_pooling_2d(h1, max_pool_window_1, stride=max_pool_stride_1)

    h2 = F.dropout(F.relu(model.l2(h1)))
    h2 = F.average_pooling_2d(h2, avg_pool_window_2, stride=avg_pool_stride_2)
    h2 = F.max_pooling_2d(h2,max_pool_window_2, stride=max_pool_stride_2)

    y = model.l3(h2)

    return y.data.argmax(axis=1)


def ensemble_predict(np_data, ensemble_model_data):
    predictions = []
    for model_data in ensemble_model_data:
        predictions.append(predict(np_data, model_data)[0])
    return Counter(predictions).most_common()[0][0]


def get_model(model_name):
    root_path = '/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/models/saved_models/'

    # load the as-cnn model and its parameters
    model_data = pickle.load(open(root_path + model_name, 'rb'))
    return model_data


def get_ensemble():
    model1 = 'binary_10,20,1,4,4,9'
    model2 = 'binary_16,20,1,6,3,9'
    model3 = 'binary_20,20,1,6,3,9'
    model4 = 'binary_20,35,1,4,2,7'
    model5 = 'binary_27,29,2,8,4,8'
    models = [model1, model2, model3, model4, model5]
    root_path = '/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/models/saved_models/'

    ensemble_model_data = []
    for model in models:
        ensemble_model_data.append(get_model(model))
    return ensemble_model_data


if __name__ == '__main__':
    pass