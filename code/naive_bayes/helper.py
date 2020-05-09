import numpy as np
import math
import random
import sklearn
from sklearn.feature_extraction import text
from sklearn import naive_bayes
from sklearn import metrics
from nltk.corpus import wordnet
from nltk.corpus import stopwords

_NLTK_stopwords = stopwords.words('english')

# Takes in a dataset and returns a training set and a test set split using rotary method.
# Can I maybe do this with an array without loss of generality? Might make integration with sklearn tools easier.
def split_rotary(data,mod):
    train = list()
    test = list()
    for index,val in enumerate(data):
        if index % mod == 0:
            test.append(val)
        else:
            train.append(val)
    return np.array(train),np.array(test)

# As before, but keeps two paired data sets in their proper orders.
def split_rotary_paired(data1,data2,mod):
    data1_train, data1_test = split_rotary(data1,mod)
    data2_train, data2_test = split_rotary(data2,mod)
    return data1_train, data2_train, data1_test, data2_test

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

# Take a message and attempt to cloak it by transposing letters in some of the words.
def cloak_transposition(text, delta=1.0):
    # This startup process could probably be generalized, using a function handle instead of the length check.

    # Split the message into words.
    words = text.split()
    # We can only transpose letters in words with at least four letters, to avoid changing the start and end.
    # Find all those words.
    longword_indexes = list()
    for i in range(len(words)):
        if len(words[i]) >= 4:
            longword_indexes.append(i)
    # Determine how many of these eligible words we're supposed to modify.
    num_replace = math.ceil(delta * len(longword_indexes))
    # Pick that many words from our eligible words.
    replace = random.sample(longword_indexes, num_replace)
    # Iterate over our chosen words, and fiddle with them.
    for index in replace:
        word = words[index]
        # Adjust the word.
        letters = list(word)
        # For now, pick a random letter that isn't at the ends, and switch it with an adjacent letter
        ind = random.randint(1,len(letters)-3)
        temp = letters[ind]
        letters[ind] = letters[ind+1]
        letters[ind+1] = temp       
        new_word = ''.join(letters)
        # Put the word back.
        words[index] = new_word
    # simplistically we can just merge back with any whitespace, but ideally we would keep the whitespace
    return ' '.join(words)

# Generalized function for word replacement attack
def cloak_replacement(text, select_func, replace_func, delta=1.0):
    # First, identify which words we're going to replace.
    words = text.split()
    replace = list()
    for i in range(len(words)):
        if select_func(words[i]):
            replace.append(i)
    # Then, replace them
    for index in replace:
        if delta < 1.0: # if we're only doing some elements, check if we skip this one
            if random.rand(0,1) >= delta:
                continue
        else:
            # We are replacing this word, use the function provided to do it
            word = words[index]
            new_word = replace_func(word)
            # Put the new word back.
            words[index] = new_word
    # Simplistically join back the words. Ideally, we would rejoin using the original whitespace.
    # TODO: reuse original whitespace somehow
    return ' '.join(words)

# Helper function to avoid replacing stopwords
def select_non_stopword(word):
    # Return true if word is NOT in the NLTK stopword list.
    return word not in _NLTK_stopwords

# Helper function for thesaurus attack
def replace_synonym(word):
    # Use NLTK wordnet to find synonyms of target word
    synonyms = list()
    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    # Remove duplicates; there will be many
    synonyms = set(synonyms)
    # Also need to remove the original word, which will be in here if the word has multiple meanings
    synonyms.discard(word.lower())
    # Pick a random synonym
    replacement = random.choice(list(synonyms)) # choice needs a list or tuple
    return replacement
