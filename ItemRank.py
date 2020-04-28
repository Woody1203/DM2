import numpy as np
import pandas as pd


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

def itemrank(data, alpha, prec):
    movie_list = []           # list of all the movies as object with initially only their id
    user_list = []            # list of all the users as object with initially only their id

    # nmovie, nuser = movie_ratings_df.shape
    nmovie, nuser = data.shape

    for m in range(nmovie):
        id = m+1
        "data change needed"
        all_users = data.T[id].to_numpy()
        movie = Movie(id, all_users)
        movie_list.append(movie)

    for u in range(nuser):
        id = u+1
        "data change needed"
        all_movies =  data[id].to_numpy()
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
        CM[:,c] /= sum(CM[:,c])

    # Prediction of movie ranking
    pred = np.zeros((nuser, nmovie))
    k = 0
    for u in user_list:
        "not normalized in algo, normalized in paper"
        d = u.all_movies
        d /= np.linalg.norm(d)
        "ones in algo, ones normalized in paper"
        IR = np.ones(nmovie) / nmovie
        converged = False
        ite = 0
        while not converged:
            ite += 1
            old_IR = IR
            IR = alpha * np.dot(CM, IR) + (1-alpha) * d
            converged = (old_IR - IR < prec).all()

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
            pred[i][j] = transform_to_ratings(maxi, mini, pred[i][j])

    return pred

def transform_to_ratings(maximum, minimum, value):
    #  f(a)=c and f(b)=d)
    # f(t) = c + (d-c)/(b-a) * (t-a)
    return int(round(1 + 4/(maximum - minimum) * (value - minimum)))

######################## test zone ############################

"This data set consists of:\
* 100,000 ratings (1-5) from 943 users on 1682 movies.\
* Each user has rated at least 20 movies. "

links_df = pd.read_csv("ml-100k/u.data", sep="\t", header = 0, names = ["userId", "movieId", "rating", "timeStamp"], index_col=False)
movie_ratings_df=links_df.pivot_table(index='movieId',columns='userId',values='rating').fillna(0)

nmovie, nuser = movie_ratings_df.shape
prediction = itemrank(movie_ratings_df, 0.85, 0.01)


print(prediction)

" to do next :" \
"- allow matrix to be passed as an argument" \
"- make the result like the ratings" \
"- check the original paper if the method is well used (normalization, alpha value, correlation matrix, etc"