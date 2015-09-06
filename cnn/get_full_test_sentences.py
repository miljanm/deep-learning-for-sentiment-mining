from csv import reader, writer
from collections import defaultdict
import numpy as np
import re
import random
from pprint import pprint
from itertools import islice
from nltk.tokenize import word_tokenize, RegexpTokenizer

# form character replacement table
rep = {"&#44": ",",
       "&#58": ";",
       "&#59": ":",
       "&#96": "`",
       "(": "-LRB-",
       ")": "-RRB-",
}

# escape and compile re pattern into a dict
rep = dict((re.escape(k), v) for k, v in rep.iteritems())
pattern = re.compile("|".join(rep.keys()))


# read the data from stanford set, and store it as sent:sentiment_list
stanford_dict = defaultdict(list)
with open('./data/stanford_data/sentlex_exp12.txt') as file1, open('./data/stanford_data/rawscores_exp12.txt') as file2:
    sent_reader = reader(file1)
    score_reader = reader(file2)
    for row1, row2 in zip(sent_reader, score_reader):
        # check dict for a all regex matches and apply transformation
        sent = pattern.sub(lambda m: rep[re.escape(m.group(0))], row1[1])
        # store as sent:sentiment
        stanford_dict[sent] = row2[1:]


# dict to hold the kaggle test data
kaggle_set = defaultdict(int)
with open('./data/kaggle_data/test.tsv') as file1:
    source_reader = reader(file1, dialect="excel-tab")
    # skip the first line
    source_reader.next()

    # for each sentence key, find the longest parse and store that as a sentence corresponding to that key
    for row in source_reader:
        key = row[1]
        content = row[2]
        if key in kaggle_set:
            if len(word_tokenize(content)) > len(word_tokenize(kaggle_set[key])):
                kaggle_set[key] = content
            else:
                continue
        else:
            kaggle_set[key] = content


result_dict = {}
not_found = 0
tokenizer = RegexpTokenizer(r'\w+')
# get sentiment for each train sentence
for entry in kaggle_set.values():
    if len(tokenizer.tokenize(entry)) > 50:
        print "Success!"
        print entry
        continue
    if entry in stanford_dict:
        scores = stanford_dict[entry]
        # calculate sentence sentiment and map it to a corresponding state
        score = sum(map(int, scores))/(len(scores) * 25.0)
        if score <= 0.2:
            score = 0
        elif score > 0.2 and score <= 0.4:
            score = 1
        elif score > 0.4 and score <= 0.6:
            score = 2
        elif score > 0.6 and score <= 0.8:
            score = 3
        else:
            score = 4
        result_dict[entry] = score
    else:
        not_found += 1
        print entry

print 'Not found: ', not_found

with open('./data/kaggle_data/kaggle_full_test.txt', 'wb') as file2:
    # write the submission
    submission_writer = writer(file2, dialect="excel-tab")
    submission_writer.writerow(['PhraseId', 'SentenceId', 'Phrase', 'Sentiment'])
    for sentence, sentiment in result_dict.items():
        submission_writer.writerow(['0', '0', sentence, sentiment])