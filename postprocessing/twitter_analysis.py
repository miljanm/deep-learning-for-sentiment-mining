import argparse
import nltk
from pprint import pprint
from preprocessing.neural_feature_design import extract_neural_features
from models.conv_net_model import predict, get_model
from gensim.models.word2vec import Word2Vec
from itertools import islice
from collections import Counter, defaultdict
import pickle
import csv, pdb
import time
import string
import matplotlib.pyplot as plt
import numpy as np
from time import mktime
from datetime import datetime


__author__ = 'miljan'


def _read_tweets():
    with open('../data/twitter/tweets_aug_tube_strike.tsv', 'rU') as file1:
        # filter removes the rows starting with # (comments)
        file_reader = csv.reader(filter(lambda x: x[0] in string.digits, file1), delimiter='\t', dialect=csv.excel_tab)

        l_tweets = []
        l_timetamps = []
        l_users = []
        for row in file_reader:
            l_timetamps.append(time.strptime(row[1], '%a %b %d %H:%M:%S +0000 %Y'))
            l_tweets.append(row[2])
            l_users.append(row[4])
    return l_tweets, l_timetamps, l_users


def _group_by_hour(data):
    hour_quantization = defaultdict(list)
    for item in data:
        dt = item[0]
        # for each (hour) append combination, add all the sentiment
        hour_quantization[(dt.year, dt.month, dt.day, dt.hour)].append(item[1])
    return hour_quantization


def plot_tweets_distribution(hours, tweets):
    hours = [dt.hour for dt in hours]
    tweets = list(tweets)

    new_x = range(24)
    LABELS = ['16', '17', '18', '19', '20', '21', '22', '23', '00', '01',
              '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
    x = range(16, 24) + range(0, 16)
    y = []
    index = 0
    for val in x:
        if val in hours:
            y.append(tweets[index])
            index += 1
        else:
            y.append(0)

    width = 0.35
    fig, ax = plt.subplots()
    # ax.set_xlim(right=50)

    ax.bar(new_x, y, width, color='b')
    plt.xticks(new_x, LABELS)
    plt.show()


def plot_sentiment():
    l_tweets, l_timestamps, l_users = _read_tweets()

    root = '/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/data/twitter_stats/'
    l_a, d_b = pickle.load(open(root + 'twitter_aug_stats.pkl', 'rb'))

    l_timestamps = [datetime.fromtimestamp(mktime(struct)) for struct in l_timestamps]
    data = zip(l_timestamps, l_a)
    data = _group_by_hour(data)

    # create datetime objects from stamps
    x = np.array([datetime(*dat) for dat in data.keys()])
    # sum the sentiment for each timestamp
    y = np.array([sum(val) for val in data.values()])
    # get the number of tweets for distribution plot

    data1 = zip(x, y)
    data1.sort(key=lambda k: k[0])
    x, y = zip(*data1)

    plt.plot(x, y, marker='o')
    plt.xlabel('Hour of the day')
    plt.ylabel('Sentiment strength')
    plt.title('Distribution of sentiment during the tube strike day')
    plt.show()


def predict_sentiment():
    l_tweets, l_timestamps, l_users = _read_tweets()

    l_overall_sentiment = []
    d_source_sentiment = defaultdict(list)

    # load word2vec model
    w2v_mod_path = '/Users/miljan/PycharmProjects/thesis-shared/data/GoogleNews-vectors-negative300.bin'
    print '\nloading w2v model...'
    w2v_model = Word2Vec.load_word2vec_format(w2v_mod_path, binary=True)  # C binary format

    counter = 0
    model = get_model('binary_10,20,1,4,4,9')

    for s_text, time, user in zip(l_tweets, l_timestamps, l_users):

        print counter
        counter += 1

        np_data = extract_neural_features(s_text, w2v_model)
        l_predictions = list(predict(np_data, model))

        # chunk the given text into sentences
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        l_sentences = tokenizer.tokenize(s_text)

        d_counts = Counter(l_predictions)

        count_positive = d_counts[3] + d_counts[4]
        count_neutral = d_counts[2]
        count_negative = d_counts[0] + d_counts[1]

        # collect stats
        l_overall_sentiment.append(count_positive-count_negative)
        d_source_sentiment[user].append(count_positive-count_negative)

    # print first 200 tweets
    print zip(l_tweets, l_overall_sentiment)[:300]
    pickle.dump((l_overall_sentiment,d_source_sentiment), open('../data/twitter_stats/twitter_aug_stats2.pkl', 'wb'))


if __name__ == '__main__':
    plot_sentiment()