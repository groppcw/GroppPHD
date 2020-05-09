import helper
import os
import numpy as np
import sklearn
from sklearn import metrics

# load ham and spam into lists
path = "/home/groppcw/landing/enron-spam/enron1"
data = list()
labels = list()
for fname in os.listdir(path+"/ham/"):
    infile = open(path+"/ham/"+fname,"r")
    try:
        data.append(infile.read())
        labels.append(1)
    except UnicodeDecodeError:
        print("Could not process file. Skipping",path+'/ham'+fname)
    infile.close()

for fname in os.listdir(path+"/spam/"):
    infile = open(path+"/spam/"+fname,"r")
    try:
        data.append(infile.read())
        labels.append(0)
    except UnicodeDecodeError:
        print("Could not process file. Skipping",path+'/ham'+fname)
    infile.close()

# Split into test and training data.
train_data, train_labels, test_data, test_labels = helper.split_rotary_paired(data, labels, 10)
# Convert this datafile to a sparse matrix.
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
clf = helper.train_naive_bayes(train_vec, list(train_labels))

# Evaluate classifier performance on test data
pred = clf.predict(test_vec)
score = sklearn.metrics.accuracy_score(test_labels, pred)
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
print(score)
