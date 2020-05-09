import pandas as pd
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
import time

class prepare_data:

    def __init__(self, filepath):
        self.filepath = filepath
        self.nmovie = 0
        self.nuser = 0

    def loaddata(self):
        self.links_df = pd.read_csv(self.filepath, sep="\t", header = 0, names = ["userId", "movieId", "rating", "timeStamp"], index_col=False)

        self.Mean = self.links_df.groupby(by="userId",as_index=False)['rating'].mean()
        self.links_df_avg = pd.merge(self.links_df,self.Mean,on='userId')
        self.links_df_avg['adg_rating']=self.links_df_avg['rating_x']-self.links_df_avg['rating_y']
        self.check = pd.pivot_table(self.links_df_avg,values='rating_x',index='userId',columns='movieId')
        self.nmovie, self.nuser = self.check.shape
        self.final = pd.pivot_table(self.links_df_avg,values='adg_rating',index='userId',columns='movieId')

        # Replacing NaN by Movie Average
        self.final_movie = self.final.fillna(self.final.mean(axis=0))

        # Replacing NaN by user Average
        self.final_user = self.final.apply(lambda row: row.fillna(row.mean()), axis=1)

    def get_similarity_with_user(self):
        # user similarity on replacing NAN by user avg
#         self.b = cosine_similarity(self.final_user)
#         np.fill_diagonal(self.b, 0 )
#         self.similarity_with_user = pd.DataFrame(self.b,index=self.final_user.index)
#         self.similarity_with_user.columns=self.final_user.index

        # user similarity on replacing NAN by user avg

        sim = self.final_user.dot(self.final_user.T)
        norms = np.array([np.sqrt(np.diagonal(sim))])
        b=(sim / norms / norms.T)
        np.fill_diagonal(b.values, 0)
        #b = cosine_similarity(final_user)
        #np.fill_diagonal(b, 0 )
        self.similarity_with_user = pd.DataFrame(b,index=self.final_user.index)
        self.similarity_with_user.columns=self.final_user.index

    def get_similarity_with_movie(self):
        # user similarity on replacing NAN by item(movie) avg
#         self.cosine = cosine_similarity(self.final_movie)
#         np.fill_diagonal(self.cosine, 0 )
#         self.similarity_with_movie = pd.DataFrame(self.cosine,index=self.final_movie.index)
#         self.similarity_with_movie.columns=self.final_user.index

        sim = self.final_movie.dot(self.final_movie.T)
        norms = np.array([np.sqrt(np.diagonal(sim))])
        b=(sim / norms / norms.T)
        np.fill_diagonal(b.values, 0)
        #b = cosine_similarity(final_user)
        #np.fill_diagonal(b, 0 )
        self.similarity_with_movie = pd.DataFrame(b,index=self.final_movie.index)
        self.similarity_with_movie.columns=self.final_movie.index

class knn(prepare_data):

    def __init__(self, nmovie, nuser, Mean, final, final_movie, similarity_with_movie, sim_user_neighbour_m, n):
        self.nmovie = nmovie
        self.nuser = nuser
        self.Mean = Mean
        self.final = final
        self.final_movie = final_movie
        self.df = similarity_with_movie
        self.n = n
        self.sim_user_30_m = sim_user_neighbour_m

    def find_n_neighbours(self):
        self.order = np.argsort(df.values, axis=1)[:, :self.n]
        self.df = self.df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
               .iloc[:n].index,
              index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)

    def find_n_neighbours(self):
        order = np.argsort(self.df.values, axis=1)[:, :self.n]
        neigh_matrix = self.df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
               .iloc[:n].index,
              index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
        return neigh_matrix

    def User_item_score(self, user, item):
        a = self.sim_user_30_m[self.sim_user_30_m.index==user].values
        b = a.squeeze().tolist()
        c = self.final_movie.loc[:,item]
        d = c[c.index.isin(b)]
        f = d[d.notnull()]
        self.avg_user = self.Mean.loc[self.Mean['userId'] == user,'rating'].values[0]
        index = f.index.values.squeeze().tolist()
        corr = self.df.loc[user,index]
        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['adg_score','correlation']
        fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)
        nume = fin['score'].sum()
        deno = fin['correlation'].sum()
        final_score = self.avg_user + (nume/deno)
        return final_score

    def pred(self):
        ## store the locations with not null values
        self.list_loc = list(self.final.stack().index)
        self.pred = np.zeros((self.nmovie,self.nuser))

        ## predict for each location with values in previous matrix
        for loc1, loc2 in self.list_loc:
            self.score = self.User_item_score(loc1, loc2)
            self.pred[loc1 - 1][loc2 - 1] = self.score

        return self.pred

def find_n_neighbours(df, n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index,
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df

## calculate rmse with package
from sklearn.metrics import mean_squared_error
def rmse(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return np.sqrt(mean_squared_error(prediction, ground_truth))

## rmse calculate manual way
def alternative_rmse(test1, test2, list_loc):
    error_value = 0
    for loc1, loc2 in list_loc:
        error_value = error_value + ((test1[loc1-1][loc2-1] - test2[loc1-1][loc2-1])**2)
    error_value_mean = error_value/99999
    root_mean_square_error = np.sqrt(error_value_mean)
    return root_mean_square_error

## mae with package
from sklearn.metrics import mean_absolute_error
def mae(prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten()
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return mean_absolute_error(prediction, ground_truth)

## mae calculate manual way
def alternative_mae(test1, test2, list_loc):
    error_value = 0
    for loc1, loc2 in list_loc:
        error_value = error_value + abs(test1[loc1-1][loc2-1] - test2[loc1-1][loc2-1])
        #print(abs(test1[loc1-1][loc2-1] - test2[loc1-1][loc2-1]))
    return error_value/99999

def knn_test(n = 30):

    data = prepare_data("ml-100k/u.data")
    data.loaddata()
    #data.get_similarity_with_user()
    data.get_similarity_with_movie()
    sim_user_30_m = find_n_neighbours(data.similarity_with_movie,n)
    test = knn(data.nmovie, data.nuser, data.Mean, data.final, data.final_movie, data.similarity_with_movie, sim_user_30_m, n)
    pred = test.pred()

    return pred, data.check, data.final

if __name__ == '__main__':

    test_start_time = time.time()

    for i in range(2, 31, 2):
        start_time = time.time()

        pred, data_check, data_final = knn_test(i)

        elapsed_time = time.time() - start_time
        print("************** knn for neighbour "+ str(i) + " Finished in : " + str(elapsed_time/60) + " minutes****************")

        ## test rmse and mae score for algorithms
        ## pred is the prediction result, data.check is the validation dataset and data_final.stack().index gives the location of datapoints with real ratings
        print('RMSE with package sklearn.metrics for neighbour size' + str(i) + ' is '  + str(rmse(pred, data_check.fillna(0).values)))
        print('RMSE without package sklearn.metrics for neighbour size' + str(i) + ' is ' + str(alternative_rmse(pred, data_check.fillna(0).values, list(data_final.stack().index))))
        print('MAE with package sklearn.metrics for neighbour size' + str(i) + ' is ' + str(mae(pred, data_check.fillna(0).values)))
        print('MAE without package sklearn.metrics for neighbour size' + str(i) + ' is ' + str(alternative_mae(pred, data_check.fillna(0).values, data_final.stack().index)))

    test_elapsed_time = time.time() - test_start_time
    print("total code finish in", test_elapsed_time, "seconds")

    ## 还剩下
    ## 1）knn 为 user based 而非 item based
    ##      1）根据 userimilarity matrix 找出对应user对应item的neighbours（必须已经看过该电影）
    ##      2）参考不同neighbour的weights给新给定item评分

    ## 2) knn 的输入， 看一下dataset和testset，输入similarity函数，输入k
    ## 3）不同similarity functions
    ## 4) baseline algorithm
    ## 5）cv函数编写

    ## 最后我这边可以提供三个算法，baseline，itembased knn, userbased knn 
