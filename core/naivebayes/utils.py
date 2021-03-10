import nltk
import os
import json
import string
import re
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

APP_NAME = 'core'
PACKAGE_NAME = 'naivebayes'

def read_json(filepath):
    """
    Reads the input *_tweets.json file and returns the tweets
    :param filepath: the path to the *_tweets.json file
    :return: a list of strings representing the tweets
    """
    tweets = []
    with open(filepath) as f:
        for line in f:
            tweet = json.loads(line)['text']
            tweets.append(tweet)
    return tweets

def train_test_split(pos_tweets, neg_tweets, split_percent):
    """
    Splits the given examples into train and test
    :param pos_tweets: all the positive tweets
    :param neg_tweets: all the negative tweets
    :param split_percent: the percentage of train set from the whole
    :return: train tweets, train labels, test tweets, test labels
    """
    n = len(pos_tweets) + len(neg_tweets)
    split = int(n * split_percent / 100)
    split = int(split / 2)
    train_pos = pos_tweets[:split]
    train_neg = neg_tweets[:split]
    test_pos = pos_tweets[split:]
    test_neg = neg_tweets[split:]

    train_tweets = train_pos + train_neg
    test_tweets = test_pos + test_neg
    train_labels = np.append(np.ones(len(train_pos)), np.zeros(len(train_neg)))
    test_labels = np.append(np.ones(len(test_pos)), np.zeros(len(test_neg)))

    return train_tweets, train_labels, test_tweets, test_labels

def tokenize_tweet(tweet):
    """
    Tokenizes a tweet
    :param tweet: A tweet
    :return: a list of tokens in the tweet
    """
    tweet_tokenizer = TweetTokenizer(preserve_case=False,
                                    strip_handles=True,
                                    reduce_len=True)
    tokenized_tweet = tweet_tokenizer.tokenize(tweet)
    return tokenized_tweet

def clean_tweet(tweet):
    """
    A function to remove: stock market tickers, retweet text, hyperlinks,
    hashtag sign, stop words, punctuations, emoticons
    :param tweet: A tweet
    :return: Cleaned, tokenized tweet
    """
    stopwords_english = stopwords.words("english")
    stemmer = PorterStemmer()
    # Set of happy emoticons
    emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
    # Set of sad emoticons
    emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
    # Union of happy and sad emoticons
    emoticons = emoticons_happy.union(emoticons_sad)
    # Remove stock market tickers eg., $AAPL
    tweet = re.sub(r'\$\w*', '', tweet)
    # Remove retweet indicator "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # Remove any hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # Remove '#' sign
    tweet = re.sub(r'#', '', tweet)
    tweet = tokenize_tweet(tweet)

    tweet_clean = []
    for word in tweet:
        # Remove stopwords, emoticons, punctuations
        if word not in stopwords_english and word not in string.punctuation:
            # Do stemming
            stemmed_word = stemmer.stem(word)
            tweet_clean.append(stemmed_word)
    return tweet_clean

def create_freq_dict(tweets, labels):
    """
    Create a frequency dictionary where the key is (word, label) and the value is its count
    :param tweets: A list of tweets
    :param labels: Labels corresponding to the tweets
    :return: A dictionary with the frequency of all pairs of (word, label)
    """
    freq_dict = dict()
    # For each tweet and its corresponding label
    for tweet, label in zip(tweets, labels):
        # Clean the tweet
        tweet_clean = clean_tweet(tweet)
        # For each word
        for word in tweet_clean:
            # Key will be (word, label) e.g., ('happy', 1), ('happy', 0)
            key = (word, label)
            # If the key already exists, increment the count by 1
            if key in freq_dict:
                freq_dict[key] += 1
            # If the key isn't there, start the counter
            else:
                freq_dict[key] = 1
    return freq_dict

def write_model_json(logprior, loglikelihood):
    """
    Writes the logprior and loglikelihood to a .json file
    :param logprior: logprior computed through training
    :param loglikelihood: loglikelihood dictionary from the training
    :return: None
    """
    d = {
        "logprior": logprior,
        "loglikelihood": loglikelihood
    }
    if os.getcwd().split('\\')[-1] == "naivebayes":
        path = 'model'
    else:
        path = os.path.join(APP_NAME, PACKAGE_NAME, 'model')
    with open(os.path.join(path, "model.json"), "w") as out_file:
        json.dump(d, out_file)

    return None

def read_model_json():
    """
    Gets the logprior and loglikelihood from model.json
    :return logprior: Pretrained logprior
    :return loglikelihood: Pretrained loglikelihood
    """
    if os.getcwd().split('\\')[-1] == "naivebayes":
        path = 'model'
    else:
        path = os.path.join(APP_NAME, PACKAGE_NAME, 'model')
    with open(os.path.join(path, "model.json"), "r") as in_file:
        d = json.load(in_file)
    logprior = d['logprior']
    loglikelihood = d['loglikelihood']

    return logprior, loglikelihood


def plot_roc_curve(fpr, tpr):
    '''
    Plots roc curve
    :param fpr: False positive rate
    :param tpr: True positive rate
    :return: None
    '''
    plt.plot(fpr, tpr, color='blue', label='ROC')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()