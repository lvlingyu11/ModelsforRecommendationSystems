import numpy as np
import pandas as pd
from numpy.random import RandomState

from RecommendationEngins.NN import NearestNeighbor
from Evaluation import Metrics

'''test the nearest neighbor recommendation system, use the MovieLens Dataset'''
file_name = '../DataSets/ml-1m/ratings.dat'


k_neighbors = 5
test_nn = NearestNeighbor(file_name, k_neighbors)
predictions, truths = test_nn.predict()
range = [1, 5]

rmse = Metrics.getRMSE(predictions, truths)
nmae = Metrics.getNMAE(predictions, truths, range)

print "rmse and nmae is: ", rmse, ", ",nmae