import helper
import os
import numpy as np
import sklearn
from sklearn import metrics

# load training data into lists
print("Loading training data.")
path = "/home/groppcw/landing/aclImdb"
train_data = list()
train_labels = list()
for fname in os.listdir(path+"/train/pos/"):
    infile = open(path+"/train/pos/"+fname,"r")
    try:
        train_data.append(infile.read())
        train_labels.append(1)
    except UnicodeDecodeError:
        print("Could not process file. Skipping",path+'/ham'+fname)
    infile.close()

for fname in os.listdir(path+"/train/neg/"):
    infile = open(path+"/train/neg/"+fname,"r")
    try:
        train_data.append(infile.read())
        train_labels.append(0)
    except UnicodeDecodeError:
        print("Could not process file. Skipping",path+'/ham'+fname)
    infile.close()

# load testing data
print("Loading testing data.")
test_data = list()
test_labels = list()
for fname in os.listdir(path+"/test/pos/"):
    infile = open(path+"/test/pos/"+fname,"r")
    try:
        test_data.append(infile.read())
        test_labels.append(1)
    except UnicodeDecodeError:
        print("Could not process file. Skipping",path+'/ham'+fname)
    infile.close()

for fname in os.listdir(path+"/test/neg/"):
    infile = open(path+"/test/neg/"+fname,"r")
    try:
        test_data.append(infile.read())
        test_labels.append(0)
    except UnicodeDecodeError:
        print("Could not process file. Skipping",path+'/ham'+fname)
    infile.close()

# Convert this datafile to a sparse matrix.
print("Vectorizing data.")
train_vec = helper.vectorize_hashing(train_data)
test_vec = helper.vectorize_hashing(test_data)

#print(len(data))
#vec_data = helper.vectorize_hashing(np.array(data))
#vec_data = helper.vectorize_tfidf(np.array(data))
#print(vec_data.shape[0])
#print(vec_data[1])
#print(labels[1])
#print(train_data.shape[0])
#print(train_labels.shape[0])
#print(train_data[0])
#print(train_labels[0])

# Train classifier
print("Training classifier.")
clf = helper.train_naive_bayes(train_vec, list(train_labels))

# Evaluate classifier performance on test data
pred = clf.predict(test_vec)
score = sklearn.metrics.accuracy_score(test_labels, pred)
print("Native accuracy:")
print(score)

# Create adversarial data
adversary_data = list()
for datum in test_data:
    modified = helper.cloak_transposition(datum)
    adversary_data.append(modified)
adversary_vec = helper.vectorize_hashing(adversary_data)

# Evaluate classifier performance on advesrarial data
pred2 = clf.predict(adversary_vec)
score2 = sklearn.metrics.accuracy_score(test_labels, pred2)
print("Accuracy against adversarial modification:")
print(score)

# Let's see how much this adversary warps the unsupervised data, which is less well separated.
print("Loading unsupervised data.")

unsup_data = list()
for fname in os.listdir(path+"/train/unsup/"):
    infile = open(path+"/train/unsup/"+fname,"r")
    try:
        unsup_data.append(infile.read())
    except UnicodeDecodeError:
        print("Could not process file. Skipping",path+'/ham'+fname)
    infile.close()

unsup_vec = helper.vectorize_hashing(unsup_data)

print("Modifying unsupervised data.")
unsup_adv_data = list()
for datum in unsup_data:
    modified = helper.cloak_transposition(datum)
    unsup_adv_data.append(modified)
unsup_adv_vec = helper.vectorize_hashing(unsup_adv_data)

unsup_pred = clf.predict(unsup_vec)
unsup_adv_pred = clf.predict(unsup_adv_vec)
score3 = sklearn.metrics.accuracy_score(unsup_pred, unsup_adv_pred)
print("Similarity between unsupervised and adversarially modified unsupervised:")
print(score3)
