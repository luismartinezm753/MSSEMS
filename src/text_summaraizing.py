import math
import operator

from scipy import stats
from utils import getTweets

w1 = 0.3
w2 = 0.3
w3 = 0.3

def kullback_leibler(distr_a,distr_b):
    return stats.entropy(distr_a,distr_b)

def logistic_increasing_func(x):
    return 1 / (1 + math.exp(x))

def coverage_score(tweet):
    return 0

def significance_score(tweet):
    return 0

def diversity_score(tweet):
    return 0

def overall_score(tweet):
    return w1 * (1 - logistic_increasing_func(coverage_score(tweet))) + w2 * logistic_increasing_func(
        significance_score(tweet)) + w3 * logistic_increasing_func(diversity_score(tweet))

tweets = getTweets('baltimore')
selected = {}
scores= {}
for id,tweet in tweets.items():
    score = overall_score(tweet.text)
    scores[id] = score

sorted_scores = sorted(scores.items(), key=operator.itemgetter(1))
sorted_scores = dict(sorted_scores[0:49])