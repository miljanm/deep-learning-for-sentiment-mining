import os
from nltk.parse import stanford


__author__ = 'miljan'


os.environ['STANFORD_PARSER'] = '/usr/local/Cellar/stanford-parser/3.5.2/libexec/jars/'
os.environ['STANFORD_MODELS'] = '/usr/local/Cellar/stanford-parser/3.5.2/libexec/jars/'

parser = stanford.StanfordParser(model_path="/usr/local/Cellar/stanford-parser/3.5.2/libexec/jars/edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")


def get_sentence_trees(sentences):
    return parser.sentiment_tree_parse(sentences)
