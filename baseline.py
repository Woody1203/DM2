import numpy as np
import pandas as pd

def baseline(A, B):
    B = B.replace(0, 'NaN')
    A = pd.pivot_table(A,values='rating',index='userId',columns='movieId')
    A.loc['mean'] = A.mean()
    A = A.fillna(0)
    # data = A.drop(['mean'])
    nmovie, nuser = B.shape
    pred = np.zeros((nmovie,nuser))
    # print(nmovie, nuser)

    list_loc = list(B.stack().index)
    # print("this is A[loc]", A.loc['mean'])

    # mark = 0

    for i,j in list_loc:
        # mark += 1
        # print("this is mark", mark)
        # print("this is i", i)
        # print("this is j", j)
        # print("this is pred[i-1][j-1]", pred[i-1][j-1])
        # print("this is A.loc['mean'][j]", A.loc['mean'][j])

        # pred[i-1][j-1] = 0

        try:
            pred[i-1][j-1] = A.loc['mean'][j]
            # print("location", i, j)
            # print(pred[i-1][j-1])
        except:
            pred[i-1][j-1] = 0
    return pred

def baseline_1(data_as_links, matrix_pred_loc):
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