import numpy as np
import logging
import pandas as pd
from numpy.random import RandomState
from sklearn.metrics.pairwise import cosine_similarity
import operator
import random
from scipy import spatial

from BaseModel import BaseModel

logger = logging.getLogger(__name__)

class NearestNeighbor():
    def __init__(self, filename, k):
        self.kneighbor = k

        header_names = ['userid', 'movieid', 'rating', 'timestamp']
        ratings = pd.read_csv(filename, sep='::', header=None, names=header_names, engine='python').as_matrix()
        self.n_user = max([x[0] for x in ratings])
        self.n_item = max([x[1] for x in ratings])
        # Only consider the users who had rated 20 or more movies
        # the dataset is as [user_id, movie_id, rating, time_stamp]
        user_movie_num = {}  # record the num of movies each user has rated
        for r in ratings:
            if r[0] in user_movie_num:
                user_movie_num[r[0]] += 1
            else:
                user_movie_num[r[0]] = 1

        filtered_ratings = []  # filter the users, select those who have rated more than 20 or equal to 20 movies
        for r in ratings:
            if user_movie_num[r[0]] >= 20:
                filtered_ratings.append(r)
        print "The original ratings len: ", len(ratings)
        print "The filtered ratings len: ", len(filtered_ratings)

        self.user_movie_rating = {}  # record the ratings each user given, x=r if user rated item, and 0 otherwise
        user_ids = []
        for fr in filtered_ratings:
            if fr[0] not in self.user_movie_rating:
                user_ids.append(fr[0])
                self.user_movie_rating[fr[0]] = [0] * self.n_item
                self.user_movie_rating[fr[0]][fr[1]-1] = fr[2]
            else:
                self.user_movie_rating[fr[0]][fr[1]-1] = fr[2]

        # Randomly select users for query set and reference set
        user_num = len(user_ids)
        print "len of selected users: ", user_num
        training_ratio = 0.3
        training_num = int(user_num * training_ratio)
        testing_num = user_num - training_num
        self.training_set = user_ids[0:training_num]
        self.testing_set = user_ids[training_num:]

        # predict the rating for testing set (In KNN, the training/fitting block just memorizing the training data)
        # self.predict()




    def predict(self):
        '''train the model with nearest neighbor - user-based rating prediction (rely on the opinion of like-minded users
        to predict a rating). There is another approach which is item-based recommendation (look at rating given to similar
        items).
        User-based NN: weigh the neighbors by its similarity to the user u. The weight is computed as the cosine similarity'''

        predictions = {}
        truths = {}
        for uid in self.testing_set:
            predictions[uid] = {}
            truths[uid] = {}
            # select specific number of movies for prediction. e.g. 15 rated
            user_row = self.user_movie_rating[uid]
            rated_indices = np.nonzero(user_row)[0]
            rated = set(rated_indices)
            non_rated = set(xrange(len(user_row))) - rated
            n_samples = 15
            sampled_rated = random.sample(rated, n_samples)  # rated item indices

            # Compute the similarity between the user and user in training set
            # select the top k similar neighbors, these neighbors should rated the selected item
            for sr in sampled_rated:
                similarity = self.calcSimilarity(uid, sr)
                print similarity
                sorted_similarity = sorted(similarity.items(), key=operator.itemgetter(1)) # result is a list of tuples with second element sorted
                top_k_indices = [x[0] for x in sorted_similarity][0:self.kneighbor]  # x[0] is user id, x[1] is similarity value
                top_k_weights = [x[1] for x in sorted_similarity][0:self.kneighbor]
                # Compute the prediction value
                weight_total = 0
                for w in top_k_weights:
                    weight_total += abs(w)
                predict_score = 0
                idx = 0
                while idx < self.kneighbor and idx < len(top_k_indices):
                    predict_score += top_k_weights[idx]*self.user_movie_rating[top_k_indices[idx]][sr-1]
                    idx += 1
                predict_score = predict_score/weight_total
                predictions[uid][sr] = predict_score
                truths[uid][sr] = self.user_movie_rating[uid][sr-1]
        return predictions, truths


    def calcSimilarity(self, user_id, rated_sample_id):
        similarity = {}
        for u in self.training_set:
            rated = np.nonzero(self.user_movie_rating[u])[0]
            if rated_sample_id in rated:
                similarity[u] = 1-spatial.distance.cosine(self.user_movie_rating[user_id], self.user_movie_rating[u])
                # similarity[u] = cosine_similarity(self.user_movie_rating[user_id], self.user_movie_rating[u])[0][0]  # passing 1d array as input is deprecated in sklearn version 0.17, and will raise value in 0.19
                print "similarity u: ", similarity[u]
        return similarity