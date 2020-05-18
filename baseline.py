import numpy as np
import pandas as pd


def baseline(data_as_links, matrix_pred_loc):
    data_as_links = pd.pivot_table(data_as_links, values='rating', index='userId', columns='movieId')
    data_as_links.loc['mean'] = data_as_links.mean()
    data_as_links = data_as_links.drop(['mean'])
    nmovie, nuser = data_as_links.shape
    pred = np.zeros((nmovie,nuser))

    list_loc = list(matrix_pred_loc.stack().index)
    for i,j in list_loc:
        pred[i-1][j-1] = data_as_links.loc['mean'][j]

    return pred

def baseline_2(ratings_matrix):
    ratings_matrix_copy = pd.DataFrame.copy(ratings_matrix)
    nuser, nmovie = ratings_matrix_copy.shape

    for i in range(nmovie):
        ratings_matrix_copy[i+1] = ratings_matrix_copy[i+1].mean()

    return ratings_matrix_copy

if __name__ == '__main__':
    baseline()