import numpy as np
import pandas as pd

def baseline(A, B):
    B = B.replace(0, 'NaN')
    A = pd.pivot_table(A,values='rating',index='userId',columns='movieId')
    A.loc['mean'] = A.mean()
    A = A.fillna(0)
    nmovie, nuser = B.shape
    pred = np.zeros((nmovie,nuser))
    list_loc = list(B.stack().index)
    for i,j in list_loc:
        try:
            pred[i-1][j-1] = A.loc['mean'][j]
        except:
            pred[i-1][j-1] = 0
    return pred


if __name__ == '__main__':
    baseline()