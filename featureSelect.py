
import numpy as np
from sklearn.cross_validation import cross_val_score
import sys


# This code was adapted from Sebastian Raschka
def forward_search(nfeatures, max_k, criterion_func, params):
    '''
    Forward search for best subset of features
    :param features: list of indeces indicating features
    :param max_k:
    :param criterion_func:
    :param params:
    :return:
    '''


    features = list(xrange(0, nfeatures))
    clf = params[0]
    X = params[1]
    Y = params[2]


    # Initialize empty feature subset
    feat_sub = []
    k = 0
    d = len(features)
    if max_k > d:
        max_k = d

    print('Starting Feature select:')
    while True:
        # Evaluate criteriafunc on the subset of parameters
        crit_func_max = criterion_func(feat_sub + [features[0]], clf, X, Y)
        best_feat = features[0]
        for x in features[1:]:
            crit_func_eval = criterion_func(feat_sub + [x], clf, X, Y)
            if crit_func_eval > crit_func_max:
                crit_func_max = crit_func_eval
                best_feat = x
        sys.stdout.write('.')
        feat_sub.append(best_feat)
        features.remove(best_feat)

        # Termination condition
        k = len(feat_sub)
        if k == max_k:
            break

    return feat_sub


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
