import numpy as np
import pandas as pd
from numpy import loadtxt

def normalizeRatings(Y, R):
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return(Ynorm, Ymean)

def load_precalc_params_small():

    file = open('./new_data/small_movies_X_recentes.csv', 'rb')
    X = loadtxt(file, delimiter = ",")

    file = open('./new_data/small_movies_W_recentes.csv', 'rb')
    W = loadtxt(file,delimiter = ",")

    file = open('./new_data/small_movies_b_recentes.csv', 'rb')
    b = loadtxt(file,delimiter = ",")
    b = b.reshape(1,-1)
    num_movies, num_features = X.shape
    num_users,_ = W.shape
    return(X, W, b, num_movies, num_features, num_users)
    
def load_ratings_small():
    file = open('./new_data/small_movies_Y_recentes.csv', 'rb')
    Y = loadtxt(file,delimiter = ",")

    file = open('./new_data/small_movies_R_recentes.csv', 'rb')
    R = loadtxt(file,delimiter = ",")
    return(Y,R)

def load_Movie_List_pd():
    """ returns df with and index of movies in the order they are in in the Y matrix """
    df = pd.read_csv('./new_data/small_movie_list_ptbr.csv', header=0, index_col=0,  delimiter=',', quotechar='"')
    mlist = df["title_ptbr"].to_list()
    return(mlist, df)





