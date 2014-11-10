from featureGen import FeatureGen
from featureSelect import forward_search
from featureSelect import cv_accuracy
from matplotlib import pyplot
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression

# Analyze mean force for each sensor
def analyzeSensorMeans(X, Y):
    print np.mean(X, axis=0)


# Analyze mean force for PD and nonPD subjects
def analyzeGlobalMeans(X, Y):
    a = zip(X, Y)
    nonPD = [x[1] for x, y in a if y == 0]
    PD = [x[1] for x, y in a if y == 1]
    nonPDmean, PDmean = np.mean(nonPD), np.mean(PD)
    nonPDvar, PDvar = np.var(nonPD), np.var(PD)
    print "Mean: {}, variance: {}".format(nonPDmean, nonPDvar)
    print "Mean: {}, variance: {}".format(PDmean, PDvar)


# Run Logistic Regression and plot train and test errors
def plotTrainTest(clf, X, Y):
    trainingSizes = range(100, 250, 25)
    trainErrors, testErrors = [], []
    for i in trainingSizes:
        trainX, testX = X[:i], X[i:]
        trainY, testY = Y[:i], Y[i:]
        clf.fit(trainX, trainY)

        # Training error
        output = clf.predict(trainX)
        numWrong = sum([int(predicted != actual) for predicted, actual in zip(output, trainY)])
        numTotal = float(len(trainY))
        trainErrors.append(numWrong / numTotal)

        # Test error
        output = clf.predict(testX)
        numWrong = sum([int(predicted != actual) for predicted, actual in zip(output, testY)])
        numTotal = float(len(testY))
        testErrors.append(numWrong / numTotal)

    pyplot.plot(trainingSizes, trainErrors, label="Training error")
    pyplot.plot(trainingSizes, testErrors, label="Test error")
    pyplot.legend()
    pyplot.xlabel('Training set size')
    pyplot.ylabel('Error')
    pyplot.title('{} train and test error vs. training set size'.format(clf.__class__.__name__))
    pyplot.show()


def crossValidate(clf, X, Y):
    scores = cross_val_score(clf, X, Y, cv=5)
    accuracy = sum(scores) / len(scores)
    print "Test accuracy for {}: {}, nFeatures: {}".format(clf.__class__.__name__, accuracy, len(X[0]))
    return accuracy


def main():
    # Create feature and label vectors
    f = FeatureGen()
    X, Y = f.getXY()

    nF = len(X[0])
    subsetFeatures = forward_search(list(xrange(nF)), 10, cv_accuracy, [LogisticRegression(), X, Y])
    print subsetFeatures
    # analyzeMeans(X, Y)
    plotTrainTest(LogisticRegression(), np.array(X)[:, subsetFeatures], Y)
    # crossValidate(LogisticRegression(), X, Y)


if __name__ == "__main__":
    main()
