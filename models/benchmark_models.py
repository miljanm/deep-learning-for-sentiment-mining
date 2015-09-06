from __future__ import print_function
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from preprocessing import ngram_feature_design
import gc, pdb
import bloscpack as bp
import numpy as np
import traceback
import sys


__author__ = 'miljan'


def _get_ngram_features_dataset(top_ngrams=5000, binary=True, ngram_size=1):
    # parameters
    print("Using ngram size: ", ngram_size)
    print("Number of top ngrams selected: ", top_ngrams)
    print("Binary classification: ", binary)
    print("\n")

    # read the train and test data in numpy matrices
    print("Reading and transforming the data...\n")
    np_train_x, np_train_y = ngram_feature_design.get_ngram_processed_data('../data/kaggle_data/train.tsv',
                                                                           top_ngrams=top_ngrams,
                                                                           binary=binary,
                                                                           ngram_size=ngram_size)
    np_test_x, np_test_y = ngram_feature_design.get_ngram_processed_data('../data/kaggle_data/kaggle_full_test.tsv',
                                                                         top_ngrams=top_ngrams,
                                                                         binary=binary,
                                                                         ngram_size=ngram_size)
    print(np_test_y[:200])
    return np_train_x, np_train_y, np_test_x, np_test_y


def _average_vectors(np_data):
    # rescale the data so there are no negative points
    np_data += abs(np.min(np_data)) + 0.01
    return np.squeeze(np_data.mean(axis=2))


def _sum_vectors(np_data):
    # rescale the data so there are no negative points
    np_data += abs(np.min(np_data)) + 0.01
    return np.squeeze(np_data.sum(axis=2))


def _append_vectors(np_data):
    np_result = np.zeros((np_data.shape[0], np_data.shape[1] * np_data.shape[2]))
    for i in range(len(np_data)):
        np_result[i, :] = np_data[i, :, :].flatten(order='F')
    return np_result + (abs(np.min(np_data)) + 0.01)


def _reshape_vectors(reshape_style, np_train_x, np_test_x):
    """
    Reshape the given training numpy matrices into a single matrix row per sample

    :param reshape_style: the way to reshape the matrix
        If average, all word vectors are averaged
        If sum, the word vectors are summed
        If append, the words vectors are appended to each other
    :param np_train_x: training data matrix
    :param np_test_x: testing data matrix
    :return: reshaped data matrices
    """
    if reshape_style == 'average':
        np_train_x = _average_vectors(np_train_x)
        np_test_x = _average_vectors(np_test_x)
    elif reshape_style == 'sum':
        np_train_x = _sum_vectors(np_train_x)
        np_test_x = _sum_vectors(np_test_x)
    elif reshape_style == 'append':
        np_train_x = _append_vectors(np_train_x)
        np_test_x = _append_vectors(np_test_x)
    else:
        raise ValueError('Unknown reshape_style parameter given.')

    return np_train_x, np_test_x


def _get_neural_features_dataset(reshape_style, classes):
    print("Loading neural features data...")
    # Paths to pickled data
    root = '/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/data/blp/glove/300d/' + classes + '/'
    data_path_train_x = root + 'train_x.blp'
    data_path_train_y = root + 'train_y.blp'
    data_path_test_x = root + 'test_x.blp'
    data_path_test_y = root + 'test_y.blp'

    # Load train data
    print("\nUnpickling training data...")
    np_train_x = bp.unpack_ndarray_file(data_path_train_x)
    # np_train_x = np_train_x[:np.floor(np_train_x.shape[0]/2.0), :, :]
    np_train_y = bp.unpack_ndarray_file(data_path_train_y)
    # np_train_y = np_train_y[:np.floor(np_train_y.shape[0]/2.0)]
    gc.collect()

    # Load test data
    print("\nUnpickling testing data...")
    np_test_x = bp.unpack_ndarray_file(data_path_test_x)
    np_test_y = bp.unpack_ndarray_file(data_path_test_y)
    gc.collect()

    # reshape data matrices into single row per sample matrices (3D > 2D reshaping)
    np_train_x, np_test_x = _reshape_vectors(reshape_style, np_train_x, np_test_x)

    print("\nDone")
    return np_train_x, np_train_y, np_test_x, np_test_y


def _test_models(np_train_x, np_train_y, np_test_x, np_test_y):
    try:
        print("Creating Multinomial Naive Bayes model...")
        # apply Laplacian smoothing
        nb = MultinomialNB(alpha=1)
        np_pred_y = nb.fit(np_train_x, np_train_y).predict(np_test_x)
        print("Accuracy is: ", accuracy_score(np_test_y, np_pred_y)*100, "%\n")
        print(classification_report(np_test_y, np_pred_y, digits=5))
        
        print("Creating Gaussian Naive Bayes model...")
        # apply Laplacian smoothing
        nb = GaussianNB()
        np_pred_y = nb.fit(np_train_x, np_train_y).predict(np_test_x)
        print("Accuracy is: ", accuracy_score(np_test_y, np_pred_y)*100, "%\n")
        print(classification_report(np_test_y, np_pred_y, digits=5))

        print("Creating Bernoulli Naive Bayes model...")
        # apply Laplacian smoothing
        nb = BernoulliNB(alpha=1)
        np_pred_y = nb.fit(np_train_x, np_train_y).predict(np_test_x)
        print("Accuracy is: ", accuracy_score(np_test_y, np_pred_y)*100, "%\n")
        print(classification_report(np_test_y, np_pred_y, digits=5))
        print(confusion_matrix(np_test_y, np_pred_y))

        print("Creating Logistic Regression model...")
        lr = LogisticRegression(penalty='l2')
        np_pred_y = lr.fit(np_train_x, np_train_y).predict(np_test_x)
        print("Accuracy is: ", accuracy_score(np_test_y, np_pred_y)*100, "%\n")
        print(classification_report(np_test_y, np_pred_y, digits=5))
        print(confusion_matrix(np_test_y, np_pred_y))

        print('Creating Random Forest model...')
        rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=True, warm_start=False)
        np_pred_y = rf.fit(np_train_x, np_train_y).predict(np_test_x)
        print("Accuracy is: ", accuracy_score(np_test_y, np_pred_y)*100, "%\n")
        print(classification_report(np_test_y, np_pred_y, digits=5))

        # print("Creating Gaussian SVM model...")
        # svc = SVC(kernel="rbf")
        # np_pred_y = svc.fit(np_train_x, np_train_y).predict(np_test_x)
        # print("Accuracy is: ", accuracy_score(np_test_y, np_pred_y)*100, "%\n")
        # print(classification_report(np_test_y, np_pred_y, digits=5))
        #
        # print("Creating Polynomial second order SVM model...")
        # svc = SVC(kernel="poly", degree=2)
        # np_pred_y = svc.fit(np_train_x, np_train_y).predict(np_test_x)
        # print("Accuracy is: ", accuracy_score(np_test_y, np_pred_y)*100, "%\n")
        # print(classification_report(np_test_y, np_pred_y, digits=5))
    except Exception as e:
        print(traceback.format_exc())


def _test_svms(np_train_x, np_train_y, np_test_x, np_test_y):
    try:
        print("Creating Gaussian SVM model...")
        svc = SVC(kernel="rbf")
        np_pred_y = svc.fit(np_train_x, np_train_y).predict(np_test_x)
        print("Accuracy is: ", accuracy_score(np_test_y, np_pred_y)*100, "%\n")
        print(classification_report(np_test_y, np_pred_y, digits=5))

        print("Creating linear SVM model...")
        svc = SVC(kernel="linear")
        np_pred_y = svc.fit(np_train_x, np_train_y).predict(np_test_x)
        print("Accuracy is: ", accuracy_score(np_test_y, np_pred_y)*100, "%\n")
        print(classification_report(np_test_y, np_pred_y, digits=5))

    except Exception as e:
        print(traceback.format_exc())


def night_run():
    #### 5-CLASS NGRAMS
    # get train and test data
    np_train_x, np_train_y, np_test_x, np_test_y = _get_ngram_features_dataset(top_ngrams=10000, binary=False)
    # report different model scores
    _test_models(np_train_x, np_train_y, np_test_x, np_test_y)
    print('\n\n---------------\n\n')
    
    #### 2-CLASS NGRAMS
    # get train and test data
    np_train_x, np_train_y, np_test_x, np_test_y = _get_ngram_features_dataset(top_ngrams=10000, binary=True)
    # report different model scores
    _test_models(np_train_x, np_train_y, np_test_x, np_test_y)
    print('\n\n---------------\n\n')

    
    ##### 5-CLASS CONTINUOUS
    # get train and test data
    np_train_x, np_train_y, np_test_x, np_test_y = _get_neural_features_dataset('append', 'five_classes')
    # report different model scores
    _test_models(np_train_x, np_train_y, np_test_x, np_test_y)
    print('\n\n---------------\n\n')
    
    ##### 2-CLASS CONTINUOUS
    # get train and test data
    np_train_x, np_train_y, np_test_x, np_test_y = _get_neural_features_dataset('append', 'two_classes')
    # report different model scores
    _test_models(np_train_x, np_train_y, np_test_x, np_test_y)

    ##### 5-CLASS CONTINUOUS
    # get train and test data
    np_train_x, np_train_y, np_test_x, np_test_y = _get_neural_features_dataset('average', 'five_classes')
    # report different model scores
    _test_models(np_train_x, np_train_y, np_test_x, np_test_y)
    print('\n\n---------------\n\n')

    #### 2-CLASS CONTINUOUS
    get train and test data
    np_train_x, np_train_y, np_test_x, np_test_y = _get_neural_features_dataset('average', 'two_classes')
    # report different model scores
    _test_models(np_train_x, np_train_y, np_test_x, np_test_y)

    #### 5-CLASS NGRAMS
    # get train and test data
    np_train_x, np_train_y, np_test_x, np_test_y = _get_ngram_features_dataset(top_ngrams=1000, binary=False)
    # report different model scores
    _test_svms(np_train_x, np_train_y, np_test_x, np_test_y)
    print('\n\n---------------\n\n')
    
    ### 2-CLASS NGRAMS
    get train and test data
    np_train_x, np_train_y, np_test_x, np_test_y = _get_ngram_features_dataset(top_ngrams=1000, binary=True)
    # report different model scores
    _test_svms(np_train_x, np_train_y, np_test_x, np_test_y)
    print('\n\n---------------\n\n')


    ##### 5-CLASS CONTINUOUS
    # get train and test data
    np_train_x, np_train_y, np_test_x, np_test_y = _get_neural_features_dataset('sum', 'five_classes')
    # report different model scores
    _test_svms(np_train_x, np_train_y, np_test_x, np_test_y)
    print('\n\n---------------\n\n')

    # ##### 2-CLASS CONTINUOUS
    # get train and test data
    np_train_x, np_train_y, np_test_x, np_test_y = _get_neural_features_dataset('sum', 'two_classes')
    # report different model scores
    _test_svms(np_train_x, np_train_y, np_test_x, np_test_y)


if __name__ == '__main__':
    # # get train and test data #_get_neural_features_dataset('sum', 'five_classes') #
    np_train_x, np_train_y, np_test_x, np_test_y = _get_neural_features_dataset('sum', 'two_classes')
    # # report different model scores
    _test_models(np_train_x, np_train_y, np_test_x, np_test_y)

    # # get train and test data #_get_neural_features_dataset('sum', 'five_classes') #
    np_train_x, np_train_y, np_test_x, np_test_y = _get_ngram_features_dataset(top_ngrams=12000, binary=False)
    # # report different model scores
    _test_models(np_train_x, np_train_y, np_test_x, np_test_y)
