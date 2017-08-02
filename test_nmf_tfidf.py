from RecommendationEngins import tf_idf
from RecommendationEngins import NMF

import numpy as np
import pandas as pd
from numpy.random import RandomState


#test tf-idf recommendation engin -- cosine similarity
test_tfidf = tf_idf.TFRS()
datasource = '../DataSets/papers.csv'
test_tfidf.train(datasource)
docid = 92
recnum = 3
recommendations_tfidf = test_tfidf.predict(docid, recnum)
print recommendations_tfidf



# test NMF recommendation engin
filename_nmf = '../DataSets/ml-1m/ratings.dat'  # movie lens 1 million dataset
header_names = ['userid', 'movieid', 'rating', 'timestamp']
#UserIDs range between 1 and 6040, MovieIDs range between 1 and 3952, ratings are made on 5-star scale, timestamp is represented in seconds
datasource_nmf = pd.read_csv(filename_nmf, sep='::', header = None, names=header_names, engine='python').as_matrix()
print "Total num of ratings: ", len(datasource_nmf)
print "Length of each item: ", len(datasource_nmf[0])
n_user = 6040
n_item = 3952
n_feature = 20
max_rating = 5
min_rating = 1
scale = 5

ratings = []
# the ratings are mapped from 1 ... K to interval [0,1] using t(x) = (x-1)/K-1, in movie lens dataset, K=5
for d in datasource_nmf:
    ratings.append([d[0], d[1], float(d[2]), d[3]])
print "Original ratings: ", ratings[0][2], ratings[1][2]

# ratings[:, 2] = (ratings[:, 2]-1.0)/(scale-1.0)   # This only works for ndarray, not list
for r in ratings:
    r[2] = (r[2]-1.0)/(scale-1.0)
print "Check mapping correctly: ", ratings[0][2], ratings[1][2]
print "Check type of ratings columns: ", type(ratings[0][0]), type(ratings[0][1]), type(ratings[0][2])

# get the average for each item
item_avg_rating = {}  # item_avg_rating is like: {movie_id: [rating sum, num of ratings]}
for r in ratings:
    if r[1] in item_avg_rating:
        item_avg_rating[r[1]][0] += r[2]
        item_avg_rating[r[1]][1] += 1
    else:
        item_avg_rating[r[1]] = []
        item_avg_rating[r[1]].append(r[2])
        item_avg_rating[r[1]].append(1)
for k in item_avg_rating:
    item_avg_rating[k][0] = item_avg_rating[k][0]/item_avg_rating[k][1]


# divide the data into training and validation set as 7:3 (here, we simply divide the data. Usually the data could be
# divided into several partitions and use them for cross validation)
seed = 12345
random_state = RandomState(seed)
random_state.shuffle(ratings)
end_index_train = int(len(ratings) * 0.7)
print "end_index_train: ", end_index_train
ratings_training = ratings[0:end_index_train]
ratings_validation = ratings[end_index_train:]
iterations = 15
nmf = NMF.NonnegativeMatrixFactorization(n_user, n_item, n_feature, scale, item_avg_rating, seed, max_rating, min_rating, iterations)
nmf.train(ratings_training)
pred = nmf.predict(ratings_validation)

