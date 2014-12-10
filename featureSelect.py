import copy
import numpy as np
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, roc_auc_score, auc, roc_curve
from sklearn.multiclass import OneVsRestClassifier
import sys


# This code was adapted from Sebastian Raschka
def forward_search(nfeatures, criterion_func, params):
    '''
    Forward search for best subset of features
    :param nfeatures: list of indices indicating features
    :param criterion_func:
    :param params:
    :return:
    '''

    features = range(nfeatures)
    clf, X, Y = params
    feat_sub = []
    best_feat_sub = []
    best_score = 0.

    print('Starting Feature select:')
    while len(feat_sub) < nfeatures:
        # Evaluate criteriafunc on the subset of parameters
        new_feat, new_score = max([(candidate_feat, criterion_func(feat_sub + [candidate_feat], clf, X, Y)) for candidate_feat in features], key = lambda x:x[1])
        feat_sub.append(new_feat)
        features.remove(new_feat)

        if new_score > best_score:
            best_feat_sub = copy.deepcopy(feat_sub)
            best_score = new_score 

    return best_feat_sub


#Cross_validate accuracy on subset of features indicated by indexFeatures
def cv_accuracy(indexFeatures, clf, X, Y):
    X = np.array(X)
    Y = np.array(Y)
    scores = cross_val_score(clf, X[:, indexFeatures], Y, cv=5)
    accuracy = sum(scores) / len(scores)
    return accuracy

def tst_auc(indexFeatures, clf, X, Y):
    X = np.array(X)
    Y = np.array(Y)
    scores = cross_val_score(clf,  X[:, indexFeatures], Y, cv=10, scoring='roc_auc')
    avgAuc = sum(scores)/float(len(scores))
    return avgAuc

def tst_multiclass_AUC(indexFeatures, clf, X, Y):
    # Binarize the output
    X, Y = np.array(X), np.array(Y)
    X = X[:, indexFeatures]
    Y = label_binarize(Y, classes=list(set(Y)))
    n_classes = Y.shape[1]

    # shuffle and split training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5,
                                                        random_state=0)
    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(clf)
    Y_score = classifier.fit(X_train, Y_train).predict(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(Y_test[:, i], Y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(Y_test.ravel(), Y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return roc_auc['micro']
