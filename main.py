import argparse
import nltk
import simplejson as json
import sys
from pprint import pprint
from preprocessing.neural_feature_design import extract_neural_features
from models import conv_net_model
from gensim.models.word2vec import Word2Vec
from itertools import islice
from collections import Counter, defaultdict
import string
import pickle
import numpy as np
import bloscpack as bp


__author__ = 'miljan'


def _read_json_articles():
    text = json.load(open('./data/articles/tube-strike-july-2-days.json'))
    return text

def process_news_files():
    json_document = _read_json_articles()
    l_articles = [json_document[i]['_source']['content'] for i in range(len(json_document))]
    l_article_titles = [json_document[i]['_source']['title'] for i in range(len(json_document))]
    l_article_publishers = [json_document[i]['_source']['source'] for i in range(len(json_document))]
    l_article_ids = [json_document[i]['_id'] for i in range(len(json_document))]

    d_source_sentiment = defaultdict(list)
    l_overall_sentiment = []


    # parse the arguments
    parser = argparse.ArgumentParser(description='Sentiment analysis pipeline')
    parser.add_argument('-neural', '-n', action='store_true', default=True, help='Use neural features')
    parser.add_argument('-ngram', '-g', action='store_true', default=False, help='Use n-gram features')
    parser.add_argument('-convnet', '-c', action='store_true', default=True, help='Use the convolutional neural network')
    parser.add_argument('-benchmark', '-b', action='store_true', default=False, help='Use the benchmarking models')
    args = parser.parse_args()

    # check for argument clash
    if args.ngram and args.convnet:
        raise ValueError('Invalid argument combination: convnet works only with neural features.')
    elif args.ngram and args.neural:
        raise ValueError('Only one type of features can be selected at a time.')
    elif args.convnet and args.benchmark:
        raise ValueError('Only one type of model can be selected at a time.')

    # load word2vec model
    w2v_mod_path = '/Users/miljan/PycharmProjects/thesis-shared/data/GoogleNews-vectors-negative300.bin'
    print '\nloading w2v model...'
    w2v_model = Word2Vec.load_word2vec_format(w2v_mod_path, binary=True)  # C binary format

    counter = 0
    for s_text, title, publisher, a_id in zip(l_articles, l_article_titles, l_article_publishers, l_article_ids):
        print counter
        counter += 1
        title = filter(lambda x: x in string.printable, title)
        a_id = filter(lambda x: x in string.printable, a_id)
        publisher = filter(lambda x: x in string.printable, publisher)

        # create the feature matrix
        if args.neural:
            np_data = extract_neural_features(s_text, w2v_model)
        elif args.ngram:
            pass
        else:
            raise ValueError('Unknown feature type.')

        # get the model prediction
        if args.convnet:
            l_predictions = list(predict(np_data))

        # chunk the given text into sentences
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        l_sentences = tokenizer.tokenize(s_text)

        # print 'Text ID: ', a_id
        # print 'Title: ', title
        # print 'Source: ', publisher
        # print '\nIndividual sentences and sentiment:\n'

        # pprint(zip(l_sentences, l_predictions))
        d_counts = Counter(l_predictions)

        count_positive = d_counts[3] + d_counts[4]
        count_neutral = d_counts[2]
        count_negative = d_counts[0] + d_counts[1]
        # print '\n---------------'
        # print 'Overall sentiment: \n'
        # print 'Positive sentences: ', count_positive
        # print 'Neutral sentences: ', count_neutral
        # print 'Negative sentences: ', count_negative
        # print 'Overall sentiment of the given text is: ', count_positive-count_negative
        # print '\n\n\n\n'

        # collect stats
        l_overall_sentiment.append(count_positive-count_negative)
        d_source_sentiment[publisher].append(count_positive-count_negative)
    pickle.dump((l_overall_sentiment,d_source_sentiment), open('./stats_july.pkl', 'wb'))


if __name__ == '__main__':
    # get the models
    ensemble_model_data = conv_net_model.get_ensemble()

    # load test data
    root_path = '/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/data/blp/word2vec/two_classes/'
    s_x_test = 'test_x.blp'
    s_y_test = 'test_y.blp'

    # Load test data
    print "\nunpickling testing data..."
    x_test = bp.unpack_ndarray_file(root_path + s_x_test)
    y_test = bp.unpack_ndarray_file(root_path + s_y_test)

    # reshape data to match chainer format
    x_test = np.reshape(x_test, (x_test.shape[0], 1, 300, 43))

    print 'Predicting'

    predictions = []
    for entry in x_test:
        predictions.append(conv_net_model.ensemble_predict(entry, ensemble_model_data))

    counter = 0
    for real, pred in zip(y_test, predictions):
        if real == pred:
            counter += 1
    print counter/float(len(predictions))
