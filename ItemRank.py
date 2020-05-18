import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


class User(object):

    def __init__(self, id, all_movies):
        self.id = id
        self.all_movies = all_movies
        self.ratings_given = []
        self.n_ratings = 0

    # ratings_given[i] = [id_movie, movie_rating]
    def generate_ratings_given(self, movie_list):
        for m in range(len(self.all_movies)):
            if self.all_movies[m] > 0:
                self.ratings_given.append([movie_list[m], self.all_movies[m]])
        self.n_ratings = len(self.ratings_given)

class Movie(object):

    def __init__(self, id, all_users):
        self.id = id
        self.all_users = all_users
        self.ratings_recieved = []
        self.n_ratings = 0

    # ratings_recieved[i] = [id_user, movie_rating]
    def generate_ratings_recieved(self, user_list):
        for u in range(len(self.all_users)):
            if self.all_users[u] > 0:
                self.ratings_recieved.append([user_list[u], self.all_users[u]])
        self.n_ratings = len(self.ratings_recieved)

def itemrank(data_df, alpha, prec):
    """
    :param data: a dataframe of ratings of size (nuser, nmovie)
    :param alpha:
    :param prec: thresholf for convergence
    :return:
    """
    data_df_copy = pd.DataFrame.copy(data_df)
    movie_list = []           # list of all the movies as object with initially only their id
    user_list = []            # list of all the users as object with initially only their id

    nuser, nmovie = data_df_copy.shape

    for m in range(nmovie):
        id = m+1
        all_users = data_df_copy[id].to_numpy()    # all ratings data (recieved or not) for movie_id = id
        movie = Movie(id, all_users)
        movie_list.append(movie)

    for u in range(nuser):
        id = u+1
        all_movies =  data_df_copy.T[id].to_numpy()    # all ratings data (given or not) for user_id = id
        user = User(id, all_movies)
        user_list.append(user)

    for m in movie_list:
        m.generate_ratings_recieved(user_list)

    for u in user_list:
        u.generate_ratings_given(movie_list)

    # Correlation Matrix CM computation
    CM = np.zeros((nmovie, nmovie))
    for m in movie_list:
        for u in m.ratings_recieved:
            for mrated in u[0].ratings_given:
                if mrated[0].id != m.id:
                    CM[m.id - 1][mrated[0].id - 1] += 1

    # divide each element by the sum of elements of its column
    # this way C is a stochastic matrix
    for c in range(len(CM[0,:])):
        if (sum(CM[:,c]) != 0):
            CM[:,c] /= sum(CM[:,c])

    # Prediction of movie ranking
    pred = np.zeros((nuser, nmovie))
    k = 0
    for u in user_list:
        "not normalized in algo, normalized in paper"
        d = u.all_movies
        d_rel = d - d.mean()
        # d_rel /= np.linalg.norm(d_rel)
        "ones in algo, ones normalized in paper"
        IR = np.ones(nmovie) #/ nmovie
        converged = False
        ite = 0
        while not converged:
            ite += 1
            old_IR = IR
            IR = alpha * np.dot(CM, IR) + (1-alpha) * d_rel
            converged = ( abs(old_IR - IR) < prec ).all()

        pred[k] = IR
        k += 1

    # find the maximum and the minimum values of pred
    maxi = 0
    mini = 1
    for i in range(nuser):
        if max(pred[i]) > maxi:
            maxi = max(pred[i])
        if min(pred[i]) < mini:
            mini = min(pred[i])

    # transform the ranking values into ratings
    for i in range(nuser):
        for j in range(nmovie):
            # pred[i][j] = transform_to_ratings_to_int(maxi, mini, pred[i][j])
            pred[i][j] = transform_to_ratings_to_int(maxi, mini, pred[i][j])

    return pred

def transform_to_ratings(maximum, minimum, value):
    #  f(a)=c and f(b)=d
    # f(t) = c + (d-c)/(b-a) * (t-a)
    return 1 + 4/(maximum - minimum) * (value - minimum)

def transform_to_ratings_to_int(maximum, minimum, value):
    #  f(a)=c and f(b)=d
    # f(t) = c + (d-c)/(b-a) * (t-a)
    return int(round(1 + 4/(maximum - minimum) * (value - minimum)))

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

def full_prediction():
    "This data set consists of:\
    * 100,000 ratings (1-5) from 943 users on 1682 movies.\
    * Each user has rated at least 20 movies. "
    links_df = pd.read_csv("ml-100k/u.data", sep="\t", header = 0, names = ["userId", "movieId", "rating", "timeStamp"], index_col=False)
    movie_ratings_df=links_df.pivot_table(index='movieId',columns='userId',values='rating').fillna(0).T

    prec =  10**-4
    # some other param
    prediction = itemrank(movie_ratings_df, 0.95 , 10**-4)
    print(prediction)

    mse = compute_MSE(movie_ratings_df.to_numpy(), prediction)
    mae = compute_MAE(movie_ratings_df.to_numpy(), prediction)
    print(mse)
    print(mae)

    # some other param
    prediction = itemrank(movie_ratings_df, 0.9 , 10**-4)
    print(prediction)

    mse = compute_MSE(movie_ratings_df.to_numpy(), prediction)
    mae = compute_MAE(movie_ratings_df.to_numpy(), prediction)
    print(mse)
    print(mae)

    # some other param
    prediction = itemrank(movie_ratings_df, 0.85 , 10**-4)
    print(prediction)

    mse = compute_MSE(movie_ratings_df.to_numpy(), prediction)
    mae = compute_MAE(movie_ratings_df.to_numpy(), prediction)
    print(mse)
    print(mae)

    # some other param
    prediction = itemrank(movie_ratings_df, 0.8 , 10**-4)
    print(prediction)

    mse = compute_MSE(movie_ratings_df.to_numpy(), prediction)
    mae = compute_MAE(movie_ratings_df.to_numpy(), prediction)
    print(mse)
    print(mae)

    # some other param
    prediction = itemrank(movie_ratings_df, 0.75 , 10**-4)
    print(prediction)

    mse = compute_MSE(movie_ratings_df.to_numpy(), prediction)
    mae = compute_MAE(movie_ratings_df.to_numpy(), prediction)
    print(mse)
    print(mae)




def cross_validation():
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
        prediction = itemrank(train_df_copy, 0.95, 10**-4)

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
    cross_validation()
    # full_prediction()




" to do next :" \
"- check the original paper if the method is well used (normalization, alpha value, correlation matrix, etc" \
" make plots of mse and mae for differents values of alpha and threshold" \
"- use relative ratings" \
"- change the scoring"

