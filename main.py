import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ItemRank import itemrank
import matplotlib.pyplot as plt
from test0511 import ub_knn_test
from baseline import baseline
import time
import tikzplotlib


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

def kfold_cross_validation(k):
    "This data set consists of:\
    * 100,000 ratings (1-5) from 943 users on 1682 movies.\
    * Each user has rated at least 20 movies. "
    links_df = pd.read_csv("ml-100k/u.data", sep="\t", header = 0, names = ["userId", "movieId", "rating", "timeStamp"], index_col=False)
    # movie_ratings_df=links_df.pivot_table(index='movieId',columns='userId',values='rating').T
    movie_ratings_df=links_df.pivot_table(index='movieId',columns='userId',values='rating').fillna(0).T


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

    list_mse_baseline = []
    list_mae_baseline = []
    list_mse_itemrank = []
    list_mae_itemrank = []
    list_mse_ubknn = []
    list_mae_ubknn = []
    list_time_itemrank = []
    list_time_ubknn = []
    list_time_baseline = []
    k = []

    cv_start_time = time.time()

    for train, test in kf.split(links_df):
        print('fold ' + str(i+1))

        k.append(i+1)

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
        # test_set = test_set_links.pivot_table(index='movieId',columns='userId',values='rating').fillna(0).T
        test_set = test_set_links.pivot_table(index='movieId',columns='userId',values='rating').T
        test_set = test_set.reindex(list(range(movie_ratings_df.T.index.min(),movie_ratings_df.T.index.max()+1)),fill_value=np.NaN, axis='columns')
        test_set = test_set.reindex(list(range(movie_ratings_df.index.min(),movie_ratings_df.index.max()+1)),fill_value=np.NaN)
        test_set = test_set.astype(float)
        # print("this is test_set",test_set)


        ### Baseline algorithm prediction
        # baseline

        print("baseline start")
        start_time = time.time()
        prediction_baseline = baseline(train_set_links, test_set)
        # print(prediction_baseline)

        elapsed_time = time.time() - start_time
        print("********************************** the code finished in : " + str(elapsed_time / 60) + " minutes *********")

        # # performance evaluation on the training set
        # mse_train_baseline[i] = compute_MSE(train_set.to_numpy(), prediction_baseline)
        # mae_train_baseline[i] = compute_MAE(train_set.to_numpy(), prediction_baseline)
        # print("trainset mse result of baseline", mse_train_baseline[i])
        # print("trainset mse result of baseline", mae_train_baseline[i])
        # print()
        #
        # elapsed_time = time.time() - start_time
        # print("********************************** the code finished in : " + str(elapsed_time/60) + " minutes*********")
        #
        # performance evaluation on the test set
        mse_test_baseline[i] = compute_MSE(test_set.to_numpy(), prediction_baseline)
        mae_test_baseline[i] = compute_MAE(test_set.to_numpy(), prediction_baseline)
        list_time_baseline.append(elapsed_time)
        list_mse_baseline.append(mse_test_baseline[i])
        list_mae_baseline.append(mae_test_baseline[i])
        print("testset mse result of baseline", mse_test_baseline[i])
        print("testset mae result of baseline", mae_test_baseline[i])
        print("baseline finish")
        print()



        ### Itemrank prediction
        print("itemrank start")
        start_time = time.time()
        prediction_itemrank = itemrank(train_set_zeros, 0.6, 10**-4)
        elapsed_time = time.time() - start_time
        print("********************************** the code finished in : " + str(elapsed_time/60) + " minutes*********")
        print(prediction_itemrank)

        # performance evaluation on the training set
        mse_train_itemrank[i] = compute_MSE(train_set_zeros.to_numpy(), prediction_itemrank)
        mae_train_itemrank[i] = compute_MAE(train_set_zeros.to_numpy(), prediction_itemrank)
        print("trainset mse result of itemrank", mse_train_itemrank[i])
        print("trainset mse result of itemrank", mae_train_itemrank[i])


        # performance evaluation on the test set
        mse_test_itemrank[i] = compute_MSE(test_set.to_numpy(), prediction_itemrank)
        mae_test_itemrank[i] = compute_MAE(test_set.to_numpy(), prediction_itemrank)
        list_time_itemrank.append(elapsed_time)
        list_mse_itemrank.append(mse_test_itemrank[i])
        list_mae_itemrank.append(mae_test_itemrank[i])
        print("testset mse result of itemrank", mse_test_itemrank[i])
        print("testset mae result of itemrank", mae_test_itemrank[i])
        print("itemrank finish")
        print()


        ### User-based knn prediction

        # train_set_copy = pd.DataFrame.copy(train_set)
        # print("this is train_set_copy", train_set_copy)
        # print("this is train_set_links", train_set_links)
        # print("****************this is train_set", train_set_copy)
        # print("*****************this is test_set", test_set)
        #******************** train_set_copy or test dataset***********
        # nmovie, nuser = test_set.shape
        # print("this is the shape of the matrix", nmovie, nuser)
        # print("this is testset",test_set)
        print("ubknn start")
        # print("this is test set",test_set)
        start_time = time.time()
        prediction_ubknn, testing = ub_knn_test(train_set_links, test_set, 10)
        elapsed_time = time.time() - start_time
        print("********************************** the code finished in : " + str(elapsed_time/60) + " minutes*********")
        print(prediction_ubknn)

        # # performance evaluation on the training set
        # mse_train_ubknn[i] = compute_MSE(train_set_zeros.to_numpy(), prediction_ubknn)
        # mae_train_ubknn[i] = compute_MAE(train_set_zeros.to_numpy(), prediction_ubknn)
        # print("trainset mse result of ubknn",mse_train_ubknn[i])
        # print("trainset mae result of ubknn",mae_train_ubknn[i])
        # print()

        # performance evaluation on the test set
        mse_test_ubknn[i] = compute_MSE(test_set.to_numpy(), prediction_ubknn)
        mae_test_ubknn[i] = compute_MAE(test_set.to_numpy(), prediction_ubknn)
        list_time_ubknn.append(elapsed_time)
        list_mse_ubknn.append(mse_test_ubknn[i])
        list_mae_ubknn.append(mae_test_ubknn[i])
        print("testset mse result of ubknn", mse_test_ubknn[i])
        print("testset mse result of ubknn", mae_test_ubknn[i])
        print("ubknn finish")
        print()

        i += 1
        break


    cv_elapsed_time = time.time() - cv_start_time
    print("total code finish in", cv_elapsed_time, "seconds")


    # mean results for itemrank
    mean_mse_test_itemrank = mse_test_itemrank.mean()
    mean_mae_test_itemrank = mae_test_itemrank.mean()
    print("mean mae for itemrank", mean_mse_test_itemrank)
    print("mean mae for itemrank", mean_mae_test_itemrank)

    # mean results for baseline
    mean_mse_test_baseline = mse_test_baseline.mean()
    mean_mae_test_baseline = mae_test_baseline.mean()
    print("mean mse for baseline", mean_mse_test_baseline)
    print("mean mae for baseline", mean_mae_test_baseline)

    # mean results for ubknn
    mean_mse_test_ubknn = mse_test_ubknn.mean()
    mean_mae_test_ubknn = mae_test_ubknn.mean()
    print("mean mae for ubknn", mean_mse_test_ubknn)
    print("mean mae for ubknn", mean_mae_test_ubknn)


    ### Plots
    save_plot = True

    # mse plt
    # baseline results
    plt.plot(k, list_mse_baseline, "-b", label="MSE_baseline")
    #itemrank results
    plt.plot(k, list_mse_itemrank, "-y", label="MSE_itemrank")
    ## ubknn results
    plt.plot(k, list_mse_ubknn, "-c", label="MSE_ubknn")
    plt.xlabel('nfolds')
    plt.ylabel('MSE')
    plt.legend(loc="upper right")
    plt.title('MSE values for three algorithms')
    if save_plot:
        tikzplotlib.save('Latex/graph_mse_k.tex')
    plt.show()

    # mae plt
    plt.plot(k, list_mae_baseline, "-r", label="MAE_baseline")
    plt.plot(k, list_mae_itemrank, "-g", label="MAE_itemrank")
    plt.plot(k, list_mae_ubknn, "-m", label="MAE_ubknn")
    plt.xlabel('nfolds')
    plt.ylabel('MAE')
    plt.legend(loc="upper right")
    plt.title('MSE values for three algorithms')
    if save_plot:
        tikzplotlib.save('Latex/graph_mae_k.tex')
    plt.show()

    # running time
    plt.plot(k, list_time_baseline, "-r", label="running time baseline")
    plt.plot(k, list_time_itemrank, "-g", label="running time itemrank")
    plt.plot(k, list_time_ubknn, "-m", label="running time ubknn")
    plt.xlabel('nfolds')
    plt.ylabel('execution time (s)')
    plt.legend(loc="upper right")
    plt.title('running time for three algorithms')
    if save_plot:
        tikzplotlib.save('Latex/graph_time_k.tex')
    plt.show()



if __name__ == '__main__':
    kfold_cross_validation(10)
