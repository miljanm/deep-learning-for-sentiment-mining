#!/usr/bin/python
import cPickle as pickle
import csv
import difflib
import re
from collections import defaultdict
from time import sleep, strftime, gmtime
from TwitterSearch import *
import os, pdb, itertools


__author__ = 'miljan'


def query_twitter(keywords):
    """
    Queries the Twitter Search API with the given keywords and creates a pickle file of
    raw tweets collected for all search terms.
    """
    try:

        tso = TwitterSearchOrder()  # create a TwitterSearchOrder object
        tso.set_keywords(keywords)  # let's define all words/hashtags we would like to look for
        tso.set_include_entities(True)  # and give us all entity information

        ts = TwitterSearch(consumer_key='c4HS566Njmswz20ZMTjikGUSv',
                           consumer_secret='b7EH8Cpou9CjXR3pbfsA0Pdzli7YNvghQVXwPrSEV7y0bFhaSm',
                           access_token='205479111-sko5QEoOgWDdjeZBiEgZo8hUozjRhBIBCy4hNiM7',
                           access_token_secret='4fqX3U4UhTHoOwodNaNcvJ5mNWzff142BQWlQTPtn7jxa')

        counter = 0  # rate-limit counter
        sleep_at = 44900  # enforce delay after receiving ~45k tweets (450 calls and 100 tweets per page)
        sleep_for = (60 * 15 + 1)  # sleep for 15 mins
        all_tweets = []

        # iterate through all the tweets and pages of results
        for tweet in ts.search_tweets_iterable(tso):
            all_tweets.append(tweet)
            counter += 1  # increase counter
            if counter >= sleep_at:  # it's time to apply the delay
                counter = 0
                sleep(sleep_for)  # sleep for n secs

    except TwitterSearchException as e:
        print(e)

    finally:
        # pickle the collected tweets
        print('Number of tweets collected: %d' % len(all_tweets))
        pickle.dump(all_tweets, open('/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/data/twitter/tube_strike_tweets_' + strftime("%Y-%m-%d_%H:%M:%S", gmtime()) + '.pickle',
                                     'wb'))


def remove_non_ascii(s):
    """
    Removes all ASCII characters from a given string

    :param s: string to be cleaned
    :return cleaned string
    """
    return "".join(i for i in s if ord(i) < 128)


def remove_similar_tweets(processed_tweets, threshold=0.95):
    """
    Removes tweets which have similarity score over a given threshold

    :param processed_tweets: a dictionary of tweets with tweet details list as values
    :return a dictionary of tweets with tweet details list as values
    """
    counter = len(processed_tweets)
    # calculate similarity score for each pair of tweets
    for key1 in processed_tweets.keys():

        print('Tweets remaining to be processed: %d/%d' % (counter, len(processed_tweets)))
        counter -= 1

        score = 0
        for key2 in processed_tweets.keys():
            # flag telling if first key has already seen itself
            key_is_seen = False
            # skip itself
            if key1 == key2 and not key_is_seen:
                key_is_seen = True
                continue
            # calculate similarity
            new_score = difflib.SequenceMatcher(None, key1.lower(), key2.lower()).ratio()
            if new_score > score:
                score = new_score
        # tweets are almost the same, remove one
        if score > threshold:
            processed_tweets.pop(key1, None)

    return processed_tweets


def clean_tweets(filename, remove_similar=True):
    """
    Removes duplicates, re-tweets, whitespaces, non-ascii characters and makes tweets more uniform

    :param filename: file with pickled tweets from the Search APi
    :param remove_similar: whether to remove very similar tweets or not (!note: slows down computation significantly)
    :return: a dictionary of tweets with tweet details list as values
    """
    # load the tweets collected from twitter Search API
    tweets = pickle.load(open(filename, 'rb'))
    print('Length of uncleaned tweets is: %d' % len(tweets))

    # use dict in order match all tweets which repeat
    processed_tweets = defaultdict(object)
    for tweet in tweets:
        tweet_text = tweet['text']
        # remove non ascii chars, convert tweet to lowercase, remove leading and trailing whitespace
        tweet_text = remove_non_ascii(tweet_text.lower().strip())
        # ignore re-tweets
        if tweet_text[:2] == 'rt':
            continue
        # remove urls
        tweet_text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet_text)
        # add tweet details to save later
        processed_tweets[tweet_text] = [tweet['id_str'],
                                        tweet['created_at'],
                                        tweet_text,
                                        remove_non_ascii(tweet['user']['name']),
                                        remove_non_ascii(tweet['user']['screen_name']),
                                        ]

    # remove tweets which are too similar
    if remove_similar:
        remove_similar_tweets(processed_tweets)

    print('New cleaned tweets length is: %d' % len(processed_tweets.keys()))
    return processed_tweets


def write_tweets_to_file(tweets, filename):
    """
    Writes given collection of tweets to a tab separated excel file

    :param tweets: a dictionary of tweets with tweet details list as values
    :param filename: name of the file in which tweets should be written
    """
    with open(filename, 'wb') as file1:
            file_writer = csv.writer(file1, dialect=csv.excel_tab)
            for key in tweets:
                file_writer.writerow([x.replace('\n', '').replace('\t', '').replace('\r', '').replace('\r\n', '') for x in tweets[key]])


if __name__ == '__main__':
    query_twitter(['#tubestrike'])
    # tweets = []
    # for item in os.listdir('/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/data/twitter'):
    #     tweets.append(pickle.load(open('/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/data/twitter/' + item, 'rb')))
    # pickle.dump(tweets, open('/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/data/twitter/all_tweets.pickle', 'wb'))
    # tweets = pickle.load(open('/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/data/twitter/all_tweets.pickle', 'rb'))
    # flatten the list
    # tweets = list(itertools.chain.from_iterable(tweets))
    # pickle.dump(tweets, open('/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/data/twitter/all_tweets.pickle', 'wb'))
    # pdb.set_trace()
    # processed_tweets = clean_tweets('/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/data/twitter/all_tweets.pickle', remove_similar=False)
    # write_tweets_to_file(processed_tweets, '/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/data/twitter/tweets_aug_tube_strike.tsv')
    # pass