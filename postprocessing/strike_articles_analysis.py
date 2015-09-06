import pickle
import pdb
import operator
from pprint import pprint

__author__ = 'miljan'


def _get_stats(month):
    root = '/Users/miljan/PycharmProjects/entity-dependent-sentiment-mining/data/twitter_stats/'
    if month == 'August':
        l_a, d_b = pickle.load(open('stats_aug.pkl', 'rb'))
    elif month == 'July':
        l_a, d_b = pickle.load(open(root + 'twitter_aug_stats.pkl', 'rb'))
    else:
        raise ValueError('Invalid argument.')

    overall = 0
    for i in l_a:
        if i < 0:
            overall -= 1
        elif i > 0:
            overall += 1
        else:
            continue
    print 'Overall sentiment is: ', overall

    stats_per_source = []
    for src, scores in d_b.items():
        pos = 0
        neg = 0
        neut = 0
        for i in scores:
            if i < 0:
                neg += 1
            elif i > 0:
                pos += 1
            else:
                neut += 1
        stats_per_source.append((src, {'positive':pos, 'negative':neg, 'neutral':neut}))
    stats_per_source = sorted(stats_per_source, key=operator.itemgetter(1))
    print '\nNumber of positive, negative and neutral stories for each source (sorted on # of negative):'
    pprint(stats_per_source)


    sent_per_source = [(x[0], sum(x[1])) for x in d_b.items()]
    sorted_sent_per_source = sorted(sent_per_source, key=operator.itemgetter(1), reverse=True)
    print '\nOverall sentiment in different sources (sorted from most positive to least positive):'
    pprint(sorted_sent_per_source)

print '\n\nJuly stats'
print '----------------\n'

_get_stats('July')

# print '\nAugust stats:'
# print '----------------\n'
#
# _get_stats('August')


