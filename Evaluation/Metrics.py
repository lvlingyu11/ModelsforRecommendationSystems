from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

'''Evaluation metrics could include: Overall coverage, MAE, RMSE, reversal rate, weighted errors, and ROC sensitivity
 We use RMSE and NMAE (Normalized Mean Absolute Error)'''

def  getRMSE(predictions, truths):
    #RMSE
    mse = mean_squared_error(truths, predictions)
    rmse = math.sqrt(mse)
    return rmse

def getNMAE(predictions, truths, range):
    ''' Normalized MAE: the average of the absolute values of the difference between the real ratings and the predicted
    values divided by the ratings range'''
    mae = mean_absolute_error(truths, predictions)
    r_max = range[1]
    r_min = range[0]
    nmae = mae/(r_max-r_min)
    return nmae
