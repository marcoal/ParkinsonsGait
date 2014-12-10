import featureGen as fGen
import mlpy
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.cross_validation import cross_val_score
from classify import plotTrainTest


class Dtw(object):
    def __init__(self, n=5):
        self.neighbors = n

    def fit(self, x, y):
        self.X = np.array(x)
        self.Y = np.array(y)

    # Dynamic Time Warp InterSeries distance
    @staticmethod
    def dtw_interseries(s1, s2, squared=False):
        """
        :param s1: Time series 1 as list
        :param s2: Time series 2 as list
        :param squared: boolean. if true, distance is l2-norm. If false, l1-norm
        :return: unnormalized minimum-distance warp path between sequences
        """
        return mlpy.dtw_std(s1, s2, dist_only=True, squared=squared)

    def dist_matrix(self, train_mat, test_mat):
        """
        Computes pairwise dtw distances between training, testing series.
        :param train_mat: [m x n] each row is training example, n time points
        :param test_mat:  [l x n] each row is testing series, n time points
        :return: Distance matrix D, s.t. D_ij = dtw( train_mat(i), test_mat(j) )
        """
        return cdist(train_mat, test_mat, self.dtw_interseries)

    def predict(self, XNew):
        """
        Predict class label for each time-series in Xnew for now uses k-nearest neighbors because of suggestions of
        this paper http://alumni.cs.ucr.edu/~xxi/495.pdf
        TODO try other supervised learning that depend on distance metric
        :param Xnew:
        :return:
        """
        dist = self.dist_matrix(self.X, XNew)
        knn_idx = dist.argsort()[:, :self.n_neighbors]
        knn_labels = self.l[knn_idx]

        # Model Label
        mode_data = mode(knn_labels, axis=1)
        label = mode_data[0]
        prob = mode_data[1] / self.n_neighbors

        return label.ravel(), prob.ravel()


def main():
    f = fGen.FeatureGen()
    model = Dtw(n=1)
    for i in range(1,17):   #Predict classification for each of the sensors
        X, Y = f.getXY(f.getSensor_n, i)
        plotTrainTest(model, X, Y)

if __name__ == "__main__":
    main()
