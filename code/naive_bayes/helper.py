import numpy as np
from sklearn.feature_extraction import text
from sklearn import naive_bayes
from sklearn import metrics

# Takes in a dataset and returns a training set and a test set split using rotary method.
def split_rotary(data,mod):
    train = list()
    test = list()
    for index,val in enumerate(data):
        if index % mod == 0:
            test.append(val)
        else:
            train.append(val)
    return train,test

# As before, but keeps two paired data sets in their proper orders.
def split_rotary_paired(data1,data2,mod):
    data1_train, data1_test = split_rotary(data1,mod)
    data2_train, data2_test = split_rotary(data2,mod)
    return data1_train, data1_test, data2_train, data2_test

# These vectorizers take in an array containing strings of text, and tranform them into sparse feature matrices.
def vectorize_hashing(text_array):
    vectorizer = sklearn.feature_extraction.text.HashingVectorizer(stop_words = 'english', alternate_sign=False, n_features=2 ** 16)
    text_vectors = vectorizer.transform(text_array)
    return text_vectors

def vectorize_tfidf(text_array):
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(sublinear_tf=True,max_df=0.5,stop_words='english')
    text_vectors = vectorizer.fit_transform(text_array)
    return text_vectors

# This function takes a set of vectors and their labels and trains a naive bayes classifier on it.
def train_naive_bayes(x_train, y_train):
    clf = sklearn.naive_bayes.MultinomialNB(alpha=0.1)
    clf.fit(x_train, y_train)
    return clf
