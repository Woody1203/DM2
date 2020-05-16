import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ItemRank import itemrank

def compute_MSE(np_ratings, np_preds):
    ratings_flat = np_ratings.flatten()
    preds_flat = np_preds.flatten()
    mse_tot = 0
    nb_ratings = 0

    for i in range(len(ratings_flat)):
        # if the rating is available
        if (ratings_flat[i] > 0):
            diff = (ratings_flat[i] - preds_flat[i])**2
            mse_tot += diff
            nb_ratings += 1

    return mse_tot/nb_ratings

def compute_MAE(np_ratings, np_preds):
    ratings_flat = np_ratings.flatten()
    preds_flat = np_preds.flatten()
    mae_tot = 0
    nb_ratings = 0

    for i in range(len(ratings_flat)):
        # if the rating is available
        if (ratings_flat[i] > 0):
            diff = abs(ratings_flat[i] - preds_flat[i])
            mae_tot += diff
            nb_ratings += 1

    return mae_tot/nb_ratings

def kfold_cross_validation():
    "This data set consists of:\
    * 100,000 ratings (1-5) from 943 users on 1682 movies.\
    * Each user has rated at least 20 movies. "
    links_df = pd.read_csv("ml-100k/u.data", sep="\t", header = 0, names = ["userId", "movieId", "rating", "timeStamp"], index_col=False)
    movie_ratings_df=links_df.pivot_table(index='movieId',columns='userId',values='rating').fillna(0).T
    k = 4  # number of folds for the cv

    mse_train = np.zeros(k)
    mae_train = np.zeros(k)
    mse_test = np.zeros(k)
    mae_test = np.zeros(k)

    kf = KFold(n_splits=k)
    i = 0
    for train, test in kf.split(links_df):
        print('fold ' + str(i+1))
        train_set_links = links_df.iloc[train] # select index of the training set
        test_set_links = links_df.iloc[test]   # select index of the test set

        # training set : create the rating matrix and add the missing columns and rows; missing values are replaced by 0
        train_set = train_set_links.pivot_table(index='movieId',columns='userId',values='rating').fillna(0).T
        train_set = train_set.reindex(list(range(movie_ratings_df.T.index.min(),movie_ratings_df.T.index.max()+1)),fill_value=0, axis='columns')
        train_set = train_set.reindex(list(range(movie_ratings_df.index.min(),movie_ratings_df.index.max()+1)),fill_value=0)
        train_set = train_set.astype(float)

        # test set : create the rating matrix and add the missing columns and rows; missing values are replaced by 0
        test_set = test_set_links.pivot_table(index='movieId',columns='userId',values='rating').fillna(0).T
        test_set = test_set.reindex(list(range(movie_ratings_df.T.index.min(),movie_ratings_df.T.index.max()+1)),fill_value=0, axis='columns')
        test_set = test_set.reindex(list(range(movie_ratings_df.index.min(),movie_ratings_df.index.max()+1)),fill_value=0)
        test_set = test_set.astype(float)

        # prediction
        train_df_copy = pd.DataFrame.copy(train_set)
        prediction = itemrank(train_df_copy, 0.85 , 0.0001)

        # performance evaluation on the training set
        mse_train[i] = compute_MSE(train_set.to_numpy(), prediction)
        mae_train[i] = compute_MAE(train_set.to_numpy(), prediction)
        print(mse_train[i])
        print(mae_train[i])
        print()

        # performance evaluation on the test set
        mse_test[i] = compute_MSE(test_set.to_numpy(), prediction)
        mae_test[i] = compute_MAE(test_set.to_numpy(), prediction)
        print(mse_test[i])
        print(mae_test[i])
        print()

        i += 1

    mean_mse_train = mse_train.mean()
    mean_mae_train = mae_train.mean()
    print(mean_mse_train)
    print(mean_mae_train)

    mean_mse_test = mse_test.mean()
    mean_mae_test = mae_test.mean()
    print(mean_mse_test)
    print(mean_mae_test)

if __name__ == '__main__':
    kfold_cross_validation()
