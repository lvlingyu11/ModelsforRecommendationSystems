import numpy as np
import logging
from numpy.random import RandomState
import math

from BaseModel import BaseModel
from Exceptions import DimensionError
from RecommendationSystems.Evaluation import Metrics

logger = logging.getLogger(__name__)

class NonnegativeMatrixFactorization(BaseModel):
    '''Nonnegative Matrix Factorization: A rating matrix A is represented as the linear model X=UV plus a Gaussian distributed
    noise matrix Z, with Zij~N(0, sigma). A = UV+Z, which values in U and V should be non-negative. Goal is to find nonnegative
    matrix U and nonnegative matrix V that minimize the sum of the squared difference between A and X. In addition, many work
    would avoid overfitting through a regularized model: min(a-uv)^2 + lamda*(||U||^2+||V||^2), where lamda controls the extent
    of regularization and is usually determined by cross-validation.'''

    # Use the NMF-based EM to learn from incomplete ratings, E step: replace missing values with the current model estimate
    # M step: update model parameters by performing NMF on the filled-in matrix

    def __init__(self, n_user, n_item, n_feature, scale, item_avg_rating, seed = None, max_rating = None, min_rating = None, iters = 50, batch_size=1e5, converge = 1e-5):
        super(NonnegativeMatrixFactorization, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_feature = n_feature
        self.max_rating = float(max_rating) if max_rating is not None else None
        self.min_rating = float(min_rating) if min_rating is not None else None
        self.scale = scale
        self.bacth_size = batch_size
        self.converge = converge
        self.iters = iters
        self.item_avg_rating = item_avg_rating
        self.rating_matrix = [] # A: nxm matrix, n items and m users
        self.unrated_ij = [] # record the unknown rating: item i, user j

        self.random_state = RandomState(seed)
        self.item_feature = 0.1*self.random_state.rand(n_item, n_feature)    # U
        self.user_feature = 0.1*self.random_state.rand(n_feature, n_user)    # V

    def train(self, datasource):
        '''training models
        To speed up training, instead of performing batch training, it is possible to subdivide the data into mini-batches,
        and update the feature vectors after each mini-batch'''

        # The movie lens data set ratings file has 4 columns: user_id, movie_id, rating, time_stamp
        ratings_training = datasource
        # check the ratings dat set is valid or not
        self.check_ratings(ratings_training)

        # initialize the rating matrix A, row index is item id, col index is user id
        self.rating_matrix = []
        for idit in xrange(self.n_item):
            self.rating_matrix.append([])
        for i in xrange(len(self.rating_matrix)):
            self.rating_matrix[i] = [-1.0 for j in xrange(self.n_user)]
        # fill in the rating matrix with existing ratings
        for r in ratings_training:
            self.rating_matrix[r[1]-1][r[0]-1] = r[2]

        # check and record the unknown ratings
        for row in xrange(len(self.rating_matrix)):
            for col in xrange(len(self.rating_matrix[row])):
                if self.rating_matrix[row][col] == -1.0:
                    self.unrated_ij.append([row+1, col+1]) #[movie id, user id]
        print "unrated_ij: ", self.unrated_ij[0:3]

        # training without subdivision into min-chunks
        # iterations for EM algorithm
        for iteration in xrange(self.iters):
            logger.debug("iteration ", iteration)
            self.eStep(iteration)
            if iteration == 0:
                last_rmse = float("inf")
            self.mStep()
            train_pred = self.predict(ratings_training)
            train_truth = [x[2] for x in ratings_training]
            rmse = Metrics.getRMSE(train_pred, train_truth)
            change = abs(last_rmse-rmse)/last_rmse
            if change < self.converge:
                logger.info("converges at iteration ", iteration, ". Stopped.")
                break
            else:
                last_rmse = rmse


    def predict(self, data):
        predictions = []
        for d in data:
            pred = np.dot(self.item_feature[d[1]-1], np.transpose([x[d[0]-1] for x in self.user_feature]))
            if pred < 0:
                pred = 0
            if pred > 1:
                pred = 1
            predictions.append(pred)
        return predictions


    def eStep(self, iteration):
        '''Replace missing entries in rating matrix A, with corresponding values in the current model estimate'''
        # if at iteration 1, average rating should be used as initialization
        if iteration == 0:
            for l in self.unrated_ij:
                print "rating_matrix: ", self.rating_matrix[l[0]-1][l[1]-1], ", item_avg_rating: ", self.item_avg_rating[l[0]][0]
                self.rating_matrix[l[0]-1][l[1]-1] = self.item_avg_rating[l[0]][0]
        else:
            # The observed entries in A should be unchanged, unknown entries are replaced with current model estimation
            for l in self.unrated_ij:
                self.rating_matrix[l[0]-1][l[1]-1] = np.dot(self.item_feature[l[0]-1], np.transpose([x[l[1]-1] for x in self.user_feature]))

    def mStep(self):
        '''Update model parameter by performing NMF on the filled-in matrix.
        NMF: minimizing ||A-UV||^2, which is the Frobenius norm, s.t. Uij > 0 and Vij > 0. Here we did not add
        regularization form. Using Lagrange multiplier gives the update formulas, the formulas are given in paper:
        Daniel D. Lee and H. Sebastian Seung. Algorithm for non-negative matrix factorization.
        (https://papers.nips.cc/paper/1861-algorithms-for-non-negative-matrix-factorization.pdf)'''

        norminator_item = np.dot(self.rating_matrix, np.transpose(self.user_feature)) #A*transpose(V)
        denorminator_item = np.dot(np.dot(self.item_feature, self.user_feature), np.transpose(self.user_feature))
        for i in xrange(self.n_item):
            for j in xrange(self.n_feature):
                self.item_feature[i][j] *= (norminator_item[i][j]/denorminator_item[i][j])
        norminator_user = np.dot(np.transpose(self.item_feature), self.rating_matrix)
        denorminator_user = np.dot(np.dot(np.transpose(self.item_feature), self.item_feature), self.user_feature)
        for i in xrange(self.n_feature):
            for j in xrange(self.n_user):
                self.user_feature[i][j] *= (norminator_user[i][j]/denorminator_user[i][j])

        # Normalize the item feature U, in order to standardize while keeping the factorization unique
        for i in xrange(self.n_item):
            sum_u = 0.0
            for j in xrange(self.n_feature):
                sum_u += self.item_feature[i][j] ** 2
            self.item_feature[i] = [x/sum_u for x in self.item_feature[i]]
        # Since U is normalized, so the value of V should also be changed according to the normalization value of U
        for i in xrange(self.n_feature):
            sum_uk = 0.0
            for it in xrange in xrange(self.n_item):
                sum_uk += self.item_feature[i][it] ** 2
            sqrt_sum_uk = math.sqrt(sum_uk)
            self.user_feature[i] = [x * sqrt_sum_uk for x in self.user_feature[i]]


    def check_ratings(self, ratings):
        '''check the rating matrix: ratings must be a matrix with shape (n_sample, 4)
        '''

        for r in ratings:
            if len(r) <> 4:
                raise DimensionError("Invalid rating format: the number of column must be 4")

        #if not np.all(ratings[:, :2]>=1):
         #   raise ValueError("Invalid user id or movie id ")

        max_user_id = max([x[0] for x in ratings])
        if max_user_id > self.n_user:
            raise ValueError("User id exceeds the max user num")

        max_item_id = max([x[1] for x in ratings])
        if max_item_id > self.n_item:
            raise ValueError("Item id exceeds the max item num")