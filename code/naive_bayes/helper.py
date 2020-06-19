import numpy as np
import math
import random
import sklearn
from sklearn.feature_extraction import text
from sklearn import naive_bayes
from sklearn import metrics
from nltk.corpus import wordnet
from nltk.corpus import stopwords
impot nltk

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
select_non_stopwords = select_non_stopword

# Helper function to filter by word length
# To use non-default lengths in constructions requiring function arguments, pass in:
#   lambda x : select_by_length(x,length=VAL)
def select_by_length(word,length=4):
  return len(word) >= length
select_length = select_by_length

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
    # If there's nothing left (not in dictionary, for example) return the original word
    if len(synonyms) == 0:
        return word
    # Pick a random synonym
    replacement = random.choice(list(synonyms)) # choice needs a list or tuple
    return replacement

# Takes in a document and a classifier function, and returns the importance scores of each word in the document.
# Scores returned as a list.
# Classifier function must take in a document and return an array of class probabilities.
# Optional argument; what to replace each word with when checking its importance
def importance_scores(text, classifier_func, oov="OUT_OF_VOCABULARY"):
  # First, establish baseline prediction on original text.
  probs = classifier_func(text)
  print(probs)
  # Do note that if there are multiple max probabilities, argmax returns the first one
  max_class = np.argmax(probs)
  # Then, iterate over each word in the document, replacing it with oov and comparing the probability
  words = text.split()
  rvals = list()
  for word_id, word in enumerate(words):
    # Replace word with oov, then flatten list back into a string separated by spaces
    new_text = ' '.join(words[:word_id] + [oov] + words[(word_id+1):])
    new_probs = classifier_func(new_text)
    print(new_probs)
    # Compare probs together
    importance = 0
    new_max = np.argmax(new_probs)
    if max_class == new_max:
      importance = (probs[max_class] - new_probs[max_class])
      print("Same max class, difference in importance;",importance)
    else:
      importance = (probs[max_class] - new_probs[max_class]) + (new_probs[new_max] - probs[new_max])
      print("Different max class, sum of the importance differences;",importance)
    rvals.append(importance)
  return rvals

# Helper function to provide alternative importance values for algorithms that require them when we don't have queries.
def importance_uniform(text, val=1.0):
  words = text.split()
  rvals = list()
  for word in words:
    rvals.append(val)
  return rvals
# As above, but scores are word length divided by some value (default 1, aka no change)
#   Max word length is a good value for this argument if you need to make the importance scores cap at 1.
def importance_length(text, divisor=1.0):
  words = text.split()
  rvals = list()
  for word in words:
    rvals.append(len(word)/divisor)
  return rvals

# Attack skeleton for textfooler. Made generic enough to tweak without total rewrite.
# May be more generalizable if the part of speech (pos) checking is done inside the synonym function,
#  although it is more efficient to have it separate and keep the score. Maybe only do that if pos_func is provided?
# Note that to use a non query based importance function you'll need to pass in something like:
#     lambda x,y: importance_uniform(x)
def cloak_textfooler(text, classifier_func, importance_func = importance_scores, select_func = lambda x,y: select_non_stopword(x), synonym_func = None, candidate_filter = None, sim_func = None, sim_threshold = 0.8):
  text = text.strip() # Note that we're going to be splitting and rejoining this a lot, so all whitespace is equivalent
  # Determine importance scores for each word in the text.
  importance = importance_func(text, classifier_func)
  words = text.split()

  # Cull wordlist.
  selected_words_indexes = list()
  selected_words_importance = list()
  for word_id, word in enumerate(words):
    if select_func(word, importance[word_id]):
      selected_words_indexes.append(word_id)
      selected_words_importance.append(importance[word_id])

  # Sort wordlist by importance.
  sorted_word_indexes = [x for _,x in sorted(zip(selected_words_importance, selected_words_indexes),reverse=True)]
  
  # First, find our baseline prediction.
  current_text = ' '.join(words)
  orig_probs = classifier_func(current_text)
  orig_class = np.argmax(orig_probs)
  # MAIN LOOP:
  # For each word, in sorted order, identify synonyms, find the best candidate, and replace that word with it.
  # If we're able to change the predicted class, break. If not, keep our replacement, and keep going.
  for word_index in sorted_word_indexes:
    # Expand word into a set of candidates.
    word = words[word_index]
    candidates = synonym_func(word)

    # If we were provided a candidate filter, apply that now.
    if candidate_filter:
      filtered_candidates = list()
      for candidate in candidates:
        if candidate_filter(word,candidate):
          filtered_candidates.append(candidate)
      candidates = filtered_candidates

    # For each candidate, try replacing the word with that candidate. Remove candidate if it fails similarity test.
    final_candidates = list()
    final_class = list()
    final_probs = list()
    for candidate in candidates:
      new_text = ' '.join(words[:word_index] + [candidate] + words[word_index+1:])
      if sim_func(text, new_text) > sim_threshold:
        final_candidates.append(candidate)
        cand_probs = classifier_func(new_text)
        final_probs.append(cand_probs)
        final_class.append(np.argmax(cand_probs))

    # If no candidates remain, skip this word and keep looping.
    if len(final_candidates) == 0:
      continue

    # If there's any candidates that change class:
    #   Remove all candidates that don't.
    #   Select the best one, and make the replacement.
    #   Break the loop.

    # Select best one of remaining candidates.

    # Replace text with selected candidate.

    # Keep looping.
  # If we get to the end of the loop without breaking, return what we've got as a failure.

def select_importance_threshold(word, importance_val, importance_threshold=0.5):
  if importance_val >= importance_threshold:
    return True
  else:
    return False

# This function takes in any number of selection functions which take in a word and a value, and concatenates them
# If not all the chosen functions use the same arguments, you will need to use lambdas to wrap them
def create_composite_select_w_val(*funcs):
  # Flatten arg format if this was called recursively, or transform if it wasn't
  if isinstance(funcs[0],list):
    funcs = funcs[0]
  else:
    funcs = list(funcs)

  if len(funcs) == 1:
    return funcs[0]
  else:
    # Composite top of the list with checking everything after it
    # Specifically, AND together the results of this func with the func returned by a recursive call
    
    restfunc = create_composite_select_w_val(funcs[1:])
    thisfunc = funcs[0]
    return lambda x,y : thisfunc(x,y) and restfunc(x,y)

# Check if new sentence has same parts of speech as old one does
# Note that some parts of speech are sort of equivalent and this does not accomodate such nuance
# For example, "a" is a determinant but "one" is a number, so "a bicycle" and "one bicycle" are not considered equal
def semantic_pos_filter(original,new):
  # This approach can't compare if the number of words changed
  if len(original.split()) != len(new.split()):
    print("Sentences have differing word counts; unable to compare by part of speech.")
    print(original)
    print(new)
    return False
  # NLTK POS returns a list of tuples (word, POS)
  orig_tags = nltk.pos_tag(original.split(),tagset='universal')
  new_tags = nltk.pos_tag(new.split(),tagset='universal')
  # zip transforms these into (word,word,word),(POS,POS,POS), which is much easier to compare
  orig_pos = list(zip(*orig_tags))[1]
  new_pos = list(zip(*new_tags))[1]
  return orig_pos == new_pos

# Return list of synonyms for a word, using NLTK wordnet
# IF no synonyms are found (or word isn't recognized), this list will be empty
def expand_synonyms_NLTK(word):
  # Use NLTK wordnet to find synonyms of target word
  synonyms = list()
  for synset in wordnet.synsets(word):
    for lemma in synset.lemmas():
      synonyms.append(lemma.name())
  # Remove duplicates; there will be many
  synonyms = set(synonyms)
  # Also need to remove the original word, which will be in here if the word has multiple meanings
  synonyms.discard(word.lower())
  # Return whatever is left
  return list(synonyms)

expand_synonyms_nltk = expand_synonyms_NLTK
