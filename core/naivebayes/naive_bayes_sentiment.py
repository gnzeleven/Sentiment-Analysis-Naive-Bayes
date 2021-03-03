import numpy as np
from .utils import *

def naive_bayes_train(freq_dict, n_tweets, n_pos):
    """
    :param freq_dict: A dictionary with the frequency of all pairs of (word, label)
    :parm n_tweets: Total number of training tweets
    :param n_pos: Total number of training tweets that are positive
    :return logprior: Log prior - log(D_pos) - log(D_neg)
    :return loglikelihood:
    """
    loglikelihood = {}
    N_pos, N_neg = 0, 0

    # Calculate V
    vocab = set([key[0] for key in freq_dict.keys()])
    V = len(vocab)

    # Compute D_pos, D_neg, D
    D_pos, D = n_pos, n_tweets
    D_neg = D - D_pos

    # Compute N_pos and N_neg
    for pair in freq_dict.keys():
        if pair[1] == 1:
            N_pos += freq_dict[pair]
        else:
            N_neg += freq_dict[pair]

    # Compute logprior - log(D_pos) - log(D_neg)
    logprior = np.log(D_pos) - np.log(D_neg)

    # For each unique word, compute it's loglikelihood and store it in a
    # dict - used for prediction
    for word in vocab:
        freq_pos = freq_dict.get((word, 1), 0)
        freq_neg = freq_dict.get((word, 0), 0)
        P_W_pos = (freq_pos + 1) / (N_pos + V)
        P_W_neg = (freq_neg + 1) / (N_neg + V)
        loglikelihood[word] = np.log(P_W_pos) - np.log(P_W_neg)

    return logprior, loglikelihood

def naive_bayes_predict(tweet, logprior=None, loglikelihood=None):
    """
    :param tweet: A tweet for which prediction has to be done
    :param logprior: logprior computed through naive_bayes_train
    :param loglikelihood: loglikelihood dictionary - consisting of loglikelihood for each word
    :return: probability/score that the tweet is positive sentiment
    """
    # Load the logprior and loglikelihood if they aren't provided as args
    if logprior == None or loglikelihood == None:
        logprior, loglikelihood = read_model_json()
    # Clean the tweet - get list of cleaned, stemmed tokens
    word_list = clean_tweet(tweet)
    # Initialize probability/score to 0 and logprior to it
    p = logprior
    # For each word in the words list
    for word in word_list:
        # If the word is in loglikelihood dictionary
        if word in loglikelihood:
            # Add the loglikelihood[word] to the probability/score
            p += loglikelihood[word]
    return p

def naive_bayes_test(test_tweets, test_labels, logprior, loglikelihood):
    """
    :param test_tweets: A list of test tweets
    :param test_labels: corresponding test labels
    :param logprior: logprior computed through naive_bayes_train
    :param loglikelihood: loglikelihood dictionary - consisting of loglikelihood for each word
    :return error: test error
    :return accuracy: test accuracy
    """
    pred_labels = []
    # Iterate over all the test tweets
    for test_tweet in test_tweets:
        # Get the probability/score for the tweet
        p = naive_bayes_predict(test_tweet, logprior, loglikelihood)
        # Check if it's positive or negative based on the score
        if p > 0:
            pred_label = 1
        else:
            pred_label = 0
        pred_labels.append(pred_label)
    # Calculate the error
    error = (pred_labels != test_labels).sum() / len(test_labels)
    # Compute the accuracy
    accuracy = 1 - error
    return error, accuracy

if __name__ == '__main__':
    # Path to positive json and negative json
    pos_json_path = 'data/positive_tweets.json'
    neg_json_path = 'data/negative_tweets.json'
    # Percentage of training data
    split_percent = 80
    # Read the tweets
    pos_tweets = read_json(pos_json_path)
    neg_tweets = read_json(neg_json_path)
    # Split the tweets into train and test
    train_tweets, train_labels, test_tweets, test_labels = train_test_split(pos_tweets, neg_tweets, split_percent)
    # Get the frequency dictionary
    freq_dict = create_freq_dict(train_tweets, train_labels)
    # Compute logprior and loglikelihood
    n_train_pos = int(len(pos_tweets) * split_percent / 100)
    logprior, loglikelihood = naive_bayes_train(freq_dict, len(train_tweets), n_train_pos)
    # Save the logprior and loglikelihood as json
    write_model_json(logprior, loglikelihood)
    # Run test
    error, accuracy = naive_bayes_test(test_tweets, test_labels, logprior, loglikelihood)
    print("Test error: {}\nTest accuracy: {}".format(error, accuracy))
