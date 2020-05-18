import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ItemRank import itemrank
from test0511 import ub_knn_test
# from baseline import baseline, baseline_2
import time


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


def baseline(A, B):
    A = pd.pivot_table(A,values='rating',index='userId',columns='movieId')
    A.loc['mean'] = A.mean()
    A = A.fillna(0)
    # data = A.drop(['mean'])
    nmovie, nuser = B.shape
    pred = np.zeros((nmovie,nuser))
    # print(nmovie, nuser)

    list_loc = list(B.stack().index)
    # print("this is A[loc]", A.loc['mean'])

    mark = 0

    for i,j in list_loc:
        mark += 1
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


def kfold_cross_validation():
    "This data set consists of:\
    * 100,000 ratings (1-5) from 943 users on 1682 movies.\
    * Each user has rated at least 20 movies. "
    links_df = pd.read_csv("ml-100k/u.data", sep="\t", header = 0, names = ["userId", "movieId", "rating", "timeStamp"], index_col=False)
    movie_ratings_df=links_df.pivot_table(index='movieId',columns='userId',values='rating').fillna(0).T
    k = 4  # number of folds for the cv

    mse_train_itemrank = np.zeros(k)
    mae_train_itemrank = np.zeros(k)
    mse_test_itemrank = np.zeros(k)
    mae_test_itemrank = np.zeros(k)

    mse_train_ubknn = np.zeros(k)
    mae_train_ubknn = np.zeros(k)
    mse_test_ubknn = np.zeros(k)
    mae_test_ubknn = np.zeros(k)

    mse_train_baseline = np.zeros(k)
    mae_train_baseline = np.zeros(k)
    mse_test_baseline = np.zeros(k)
    mae_test_baseline = np.zeros(k)

    kf = KFold(n_splits=k)
    i = 0
    for train, test in kf.split(links_df):
        print('fold ' + str(i+1))

        ### Preprocessing
        train_set_links = links_df.iloc[train] # select index of the training set
        test_set_links = links_df.iloc[test]   # select index of the test set
        # print("this is links_df", links_df)

        # training set : create the ratings matrix and add the missing columns and rows;
        train_set = train_set_links.pivot_table(index='movieId',columns='userId',values='rating').T
        train_set = train_set.reindex(list(range(movie_ratings_df.T.index.min(),movie_ratings_df.T.index.max()+1)),fill_value=np.NaN, axis='columns')
        train_set = train_set.reindex(list(range(movie_ratings_df.index.min(),movie_ratings_df.index.max()+1)),fill_value=np.NaN)
        train_set = train_set.astype(float)
        train_set_zeros = train_set.fillna(0) # NaN -> 0
        # print("this is train_set", train_set)

        # test set : create the rating matrix and add the missing columns and rows; missing values are replaced by 0
        test_set = test_set_links.pivot_table(index='movieId',columns='userId',values='rating').fillna(0).T
        test_set = test_set.reindex(list(range(movie_ratings_df.T.index.min(),movie_ratings_df.T.index.max()+1)),fill_value=0, axis='columns')
        test_set = test_set.reindex(list(range(movie_ratings_df.index.min(),movie_ratings_df.index.max()+1)),fill_value=0)
        test_set = test_set.astype(float)
        # print("this is test_set",test_set)

        ### Baseline algorithm prediction
        print("baseline start")

        start_time = time.time()

        prediction_baseline = baseline(train_set_links, test_set)
        print(prediction_baseline)

        elapsed_time = time.time() - start_time
        print("********************************** the code finished in : " + str(elapsed_time / 60) + " minutes *********")

        # performance evaluation on the training set
        mse_train_baseline[i] = compute_MSE(train_set.to_numpy(), prediction_baseline)
        mae_train_baseline[i] = compute_MAE(train_set.to_numpy(), prediction_baseline)
        print("trainset mse result of baseline", mse_train_baseline[i])
        print("trainset mse result of baseline", mae_train_baseline[i])
        print()

        # performance evaluation on the test set
        mse_test_baseline[i] = compute_MSE(test_set.to_numpy(), prediction_baseline)
        mae_test_baseline[i] = compute_MAE(test_set.to_numpy(), prediction_baseline)
        print("testset mse result of baseline", mse_test_baseline[i])
        print("testset mae result of baseline", mae_test_baseline[i])
        print()
        print("baseline fini")

        ### Itemrank prediction
        train_set_zeros_copy = pd.DataFrame.copy(train_set_zeros)
        prediction_itemrank = itemrank(train_set_zeros_copy, 0.85 , 0.0001)
        print(prediction_itemrank)

        # performance evaluation on the training set
        mse_train_itemrank[i] = compute_MSE(train_set_zeros.to_numpy(), prediction_itemrank)
        mae_train_itemrank[i] = compute_MAE(train_set_zeros.to_numpy(), prediction_itemrank)
        print("trainset mse result of itemrank", mse_train_itemrank[i])
        print("trainset mse result of itemrank", mae_train_itemrank[i])
        print()

        # performance evaluation on the test set
        mse_test_itemrank[i] = compute_MSE(test_set.to_numpy(), prediction_itemrank)
        mae_test_itemrank[i] = compute_MAE(test_set.to_numpy(), prediction_itemrank)
        print("testset mse result of itemrank", mse_test_itemrank[i])
        print("testset mae result of itemrank", mae_test_itemrank[i])
        print()

        # ### User-based knn prediction
        # train_set_copy = pd.DataFrame.copy(train_set)
        # print("this is train_set_copy", train_set_copy)
        # print("this is train_set_links", train_set_links)
        # print("****************this is train_set", train_set_copy)
        # print("*****************this is test_set", test_set)
        # #******************** train_set_copy or test dataset***********
        # prediction_ubknn, testing = ub_knn_test(train_set_links, test_set, 10)
        # print("finish ubknn")
        # print(prediction_ubknn)
        #
        # # performance evaluation on the training set
        # mse_train_ubknn[i] = compute_MSE(train_set_zeros.to_numpy(), prediction_ubknn)
        # mae_train_ubknn[i] = compute_MAE(train_set_zeros.to_numpy(), prediction_ubknn)
        # print("trainset mse result of ubknn",mse_train_ubknn[i])
        # print("trainset mae result of ubknn",mae_train_ubknn[i])
        # print()
        #
        # # # performance evaluation on the test set
        # mse_test_ubknn[i] = compute_MSE(test_set.to_numpy(), prediction_ubknn)
        # mae_test_ubknn[i] = compute_MAE(test_set.to_numpy(), prediction_ubknn)
        # print("testset mse result of ubknn", mse_test_ubknn[i])
        # print("testset mse result of ubknn", mae_test_ubknn[i])
        # print()

        # ### User-based knn prediction
        # train_set_copy = pd.DataFrame.copy(train_set)
        # print(movie_ratings_df)
        # print(train_set_links)
        # prediction_ubknn, testing = ub_knn_test(train_set_links, movie_ratings_df, 10)
        # print(prediction_ubknn)
        #
        # # performance evaluation on the training set
        # mse_train_ubknn[i] = compute_MSE(train_set_zeros.to_numpy(), prediction_ubknn)
        # mae_train_ubknn[i] = compute_MAE(train_set_zeros.to_numpy(), prediction_ubknn)
        # print(mse_train_ubknn[i])
        # print(mae_train_ubknn[i])
        # print()
        #
        # # performance evaluation on the test set
        # mse_test_ubknn[i] = compute_MSE(test_set.to_numpy(), prediction_ubknn)
        # mae_test_ubknn[i] = compute_MAE(test_set.to_numpy(), prediction_ubknn)
        # print(mse_test_ubknn[i])
        # print(mae_test_ubknn[i])
        # print()

        i += 1
        break

    # mean results for baseline
    mean_mse_train_baseline = mse_train_baseline.mean()
    mean_mae_train_baseline = mae_train_baseline.mean()
    print(mean_mse_train_baseline)
    print(mean_mae_train_baseline)

    mean_mse_test_baseline = mse_test_baseline.mean()
    mean_mae_test_baseline = mae_test_baseline.mean()
    print(mean_mse_test_baseline)
    print(mean_mae_test_baseline)

    # mean results for itemrank
    mean_mse_train_itemrank = mse_train_itemrank.mean()
    mean_mae_train_itemrank = mae_train_itemrank.mean()
    print(mean_mse_train_itemrank)
    print(mean_mae_train_itemrank)

    mean_mse_test_itemrank = mse_test_itemrank.mean()
    mean_mae_test_itemrank = mae_test_itemrank.mean()
    print(mean_mse_test_itemrank)
    print(mean_mae_test_itemrank)

    # mean results for ubknn
    mean_mse_train_ubknn = mse_train_ubknn.mean()
    mean_mae_train_ubknn = mae_train_ubknn.mean()
    print(mean_mse_train_ubknn)
    print(mean_mae_train_ubknn)

    mean_mse_test_ubknn = mse_test_ubknn.mean()
    mean_mae_test_ubknn = mae_test_ubknn.mean()
    print(mean_mse_test_ubknn)
    print(mean_mae_test_ubknn)



if __name__ == '__main__':
    kfold_cross_validation()

