import numpy as np
import pandas as pd
import scipy

# import scipy.io as sio
import matplotlib.pyplot as plt
import pylab as plb
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import os
import glob
import codecs as cs


def review_to_words(raw_data):
    review_text = BeautifulSoup(raw_data).get_text()
    letters_only = re.sub('^[a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words('english'))
    remove_stops = [w for w in words if w not in stops]
    lmtzr = WordNetLemmatizer()
    lemma_words = [lmtzr.lemmatize(i) for i in words]
    return (" ".join(lemma_words))


def data_prepare(series):
    print("Cleaning and parsing the training set movie reviews...\n")
    clean_data = []
    for i in range(series.size):
        if ((i + 1) % 1000 == 0):
            print("Review %d of %d\n" % (i + 1, series.size))
        clean_data.append(review_to_words(series[i]))
    return clean_data


spamlist = os.listdir('dist/spam/.')
hamlist = os.listdir('dist/ham/.')

allspamword = []
for i in spamlist:

    word = []
    with cs.open('dist/spam/' + i, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            data = f.readlines()
            for line in data:
                word.append(line)
        except:
            pass

    allspamword.append(''.join(word).replace("\n", ""))

allhamword = []
for i in hamlist:

    word = []
    with cs.open('dist/ham/' + i, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            data = f.readlines()
            for line in data:
                word.append(line)
        except:
            pass

    allhamword.append(''.join(word).replace("\n", ""))

spam = data_prepare(pd.Series(allspamword))
ham = data_prepare(pd.Series(allhamword))

# spam and ham
spam_ham = spam + ham
vectorizer = CountVectorizer(max_features=500)
spam_ham_vec = vectorizer.fit_transform(spam_ham)
spam_ham_vec = spam_ham_vec.toarray()
voc = np.asarray(vectorizer.get_feature_names())
# label
spam_label = np.repeat(1, len(spam), axis=0).reshape(-1, 1)
ham_label = np.repeat(0, len(ham), axis=0).reshape(-1, 1)
label = np.vstack([spam_label, ham_label])
# new train data
# X_new = normalize(spam_ham_vec)
train = np.hstack([spam_ham_vec, label])
np.random.shuffle(train)

# split into train set and validation set
X = train[..., range(train.shape[1] - 1)]
y = train[..., -1]
X_train = X[:18000, ]
X_val = X[18000:, ]
y_train = y[:18000, ]
y_val = y[18000:, ]

BASE_DIR = './'
TEST_DIR = 'test/'
NUM_TEST_EXAMPLES = 10000
testlist = [str(x) + '.txt' for x in range(NUM_TEST_EXAMPLES)]

test = []
for i in testlist:

    word = []
    with cs.open('dist/test/' + i, 'r', encoding='utf-8', errors='ignore') as f:
        try:
            data = f.readlines()
            for line in data:
                word.append(line)
        except:
            pass

    test.append(''.join(word).replace("\n", ""))

test = data_prepare(pd.Series(test))
new_vect = CountVectorizer(vocabulary=voc)
test_spam = new_vect.fit_transform(test)
test_spam = test_spam.toarray()

data = {'training_data' :X, 'training_labels':y, 'test_data': test_spam,'feature':list(voc)}
scipy.io.savemat('spam.mat',data)




