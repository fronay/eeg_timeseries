import pandas as pd
import numpy as np
from sklearn import tree, preprocessing, svm
from sklearn.cross_validation import train_test_split, cross_val_score, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier
# previous script (daisy-chained)
from preprocess import *

# -- transform list of arrays into numpy array:

# fix length problem for peak detection array in a slightly roundabout way:
# (trust me, the 'elegant way' ended up as a nightmare
DEL_FROM_CLASS = [ind for ind, a in enumerate(PEAK_FOURIER) if a.shape != (10,) ]
# FULL_CLASS = [good for ind, good in enumerate(FULL_CLASS) if ind not in DEL_FROM_CLASS]
PEAK_FOURIER = [a for a in PEAK_FOURIER if a.shape == (10,)]

# X = np.stack(FULL_FOURIER)
X = np.stack(FULL_FOURIER)
y = np.stack(FULL_CLASS)

# print len(FULL_RAW), len(FULL_FOURIER), len(FULL_CLASS)
# split into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

# print [len(i) for i in (X_train, y_train)]
# print type(X_train[0])

svn_clf = svm.SVC(verbose=True)
svn_clf.fit(X_train, y_train)
print "svm training set result", svn_clf.score(X_test, y_test)

#print "svn forest test set result", svn_clf.score(test, test_target)


clf = RandomForestClassifier(100)
clf.fit(X_train, y_train)
print "randomforest result", clf.score(X_test, y_test)


