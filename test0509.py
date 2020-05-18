import pandas as pd
import numpy as np
#from sklearn.metrics.pairwise import cosine_similarity
import time

# class prepare_data:
#
#     def __init__(self, filepath):
#         self.filepath = filepath
#         self.nmovie = 0
#         self.nuser = 0
#
#     def loaddata(self):
#         ## load data
#         self.links_df = pd.read_csv(self.filepath, sep="\t", header = 0, names = ["userId", "movieId", "rating", "timeStamp"], index_col=False)
#         ## mean values for each position according to each user
#         self.Mean = self.links_df.groupby(by="userId",as_index=False)['rating'].mean()
#         ## calculate how much the real rating is greater mean rating for each user
#         self.links_df_avg = pd.merge(self.links_df,self.Mean,on='userId')
#         self.links_df_avg['adg_rating']=self.links_df_avg['rating_x']-self.links_df_avg['rating_y']
#         ## create pivot table for real taring
#         self.check = pd.pivot_table(self.links_df_avg,values='rating_x',index='userId',columns='movieId')
#         self.nmovie, self.nuser = self.check.shape
#         ## create pivot table for relative rating
#         self.final = pd.pivot_table(self.links_df_avg,values='adg_rating',index='userId',columns='movieId')
#
#         # Replacing NaN by Movie Average
#         self.final_movie = self.final.fillna(self.final.mean(axis=0))
#
#         # Replacing NaN by user Average
#         self.final_user = self.final.apply(lambda row: row.fillna(row.mean()), axis=1)
#
#     def get_similarity_with_user(self):
#         # user similarity on replacing NAN by user avg
# #         self.b = cosine_similarity(self.final_user)
# #         np.fill_diagonal(self.b, 0 )
# #         self.similarity_with_user = pd.DataFrame(self.b,index=self.final_user.index)
# #         self.similarity_with_user.columns=self.final_user.index
#
#         # user similarity on replacing NAN by user avg
#
#         sim = self.final_user.dot(self.final_user.T)
#         norms = np.array([np.sqrt(np.diagonal(sim))])
#         b=(sim / norms / norms.T)
#         np.fill_diagonal(b.values, 0)
#         #b = cosine_similarity(final_user)
#         #np.fill_diagonal(b, 0 )
#         self.similarity_with_user = pd.DataFrame(b,index=self.final_user.index)
#         self.similarity_with_user.columns=self.final_user.index
#
#     def get_similarity_with_movie(self):
#         # user similarity on replacing NAN by item(movie) avg
# #         self.cosine = cosine_similarity(self.final_movie)
# #         np.fill_diagonal(self.cosine, 0 )
# #         self.similarity_with_movie = pd.DataFrame(self.cosine,index=self.final_movie.index)
# #         self.similarity_with_movie.columns=self.final_user.index
#
#         sim = self.final_movie.dot(self.final_movie.T)
#         norms = np.array([np.sqrt(np.diagonal(sim))])
#         b=(sim / norms / norms.T)
#         np.fill_diagonal(b.values, 0)
#         #b = cosine_similarity(final_user)
#         #np.fill_diagonal(b, 0 )
#         self.similarity_with_movie = pd.DataFrame(b,index=self.final_movie.index)
#         self.similarity_with_movie.columns=self.final_movie.index

class knn():

    def __init__(self, trainning_data, testing_data, n):
        # self.filepath = filepath
        self.n = n
        self.links_df = trainning_data
        ## locations of the predicted datapoints
        self.list_loc = list(testing_data.stack().index)
    # def __init__(self, nmovie, nuser, Mean, final, final_movie, similarity_with_movie, sim_user_neighbour_m, n):
        # self.nmovie = nmovie
        # self.nuser = nuser
        # self.Mean = Mean
        # self.final = final
        # self.final_movie = final_movie
        # self.df = similarity_with_movie
        # self.n = n
        # self.sim_user_30_m = sim_user_neighbour_m

    def loaddata(self):
        ## load data
        ## mean values for each position according to each user
        self.Mean = self.links_df.groupby(by="userId",as_index=False)['rating'].mean()
        ## calculate how much the real rating is greater mean rating for each user
        self.links_df_avg = pd.merge(self.links_df,self.Mean,on='userId')
        self.links_df_avg['adg_rating']=self.links_df_avg['rating_x']-self.links_df_avg['rating_y']
        ## create pivot table for real taring
        self.check = pd.pivot_table(self.links_df_avg,values='rating_x',index='userId',columns='movieId')
        self.nmovie, self.nuser = self.check.shape
        ## create pivot table for relative rating
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


    # def find_n_neighbours(self, data_matrix):
    #     self.order = np.argsort(data_matrix.values, axis=1)[:, :self.n]
    #     neigh_matrix = data_matrix.apply(lambda x: pd.Series(x.sort_values(ascending=False)
    #            .iloc[:n].index,
    #           index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    #     return neigh_matrix



    def User_item_score(self, user, item):

        a = self.sim_user_30_m[self.sim_user_30_m.index==user].values
        b = a.squeeze().tolist()
        c = self.final_movie.loc[:,item]
        d = c[c.index.isin(b)]
        f = d[d.notnull()]
        self.avg_user = self.Mean.loc[self.Mean['userId'] == user,'rating'].values[0]
        index = f.index.values.squeeze().tolist()
        corr = self.similarity_with_movie.loc[user,index]
        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['adg_score','correlation']
        fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)
        nume = fin['score'].sum()
        deno = fin['correlation'].sum()
        final_score = self.avg_user + (nume/deno)
        return final_score

    def pred(self):
        ## store the locations with not null values
        ## for cross validation, here is the code to find the location to predict, maybe change it to test dataset's locations
        #list_loc = list(self.final.stack().index)
        self.pred = np.zeros((self.nmovie,self.nuser))
        self.sim_user_30_m = find_n_neighbours(self.similarity_with_movie, self.n)
        ## predict for each location with values in previous matrix
        for loc1, loc2 in self.list_loc:
            self.score = self.User_item_score(loc1, loc2)
            self.pred[loc1 - 1][loc2 - 1] = self.score

        return self.pred


# ###############################
#     def find_n_neighbours(self):
#         order = np.argsort(df.values, axis=1)[:, :n]
#         df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
#                .iloc[:n].index,
#               index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
#         return df

## calculate rmse with package
# from sklearn.metrics import mean_squared_error
# def rmse(prediction, ground_truth):
#     prediction = prediction[ground_truth.nonzero()].flatten()
#     ground_truth = ground_truth[ground_truth.nonzero()].flatten()
#     return np.sqrt(mean_squared_error(prediction, ground_truth))

def find_n_neighbours(df, n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index,
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df

## rmse calculate manual way
def alternative_rmse(test1, test2):
    list_loc = test2.stack().index
    test2 = test2.fillna(0).values
    error_value = 0
    for loc1, loc2 in list_loc:
        error_value = error_value + ((test1[loc1-1][loc2-1] - test2[loc1-1][loc2-1])**2)
    error_value_mean = error_value/99999
    root_mean_square_error = np.sqrt(error_value_mean)
    return root_mean_square_error

## mae with package
# from sklearn.metrics import mean_absolute_error
# def mae(prediction, ground_truth):
#     prediction = prediction[ground_truth.nonzero()].flatten()
#     ground_truth = ground_truth[ground_truth.nonzero()].flatten()
#     return mean_absolute_error(prediction, ground_truth)

## mae calculate manual way
def alternative_mae(test1, test2):
    #print(type(test2))
    list_loc = test2.stack().index
    test2 = test2.fillna(0).values
    error_value = 0
    for loc1, loc2 in list_loc:
        error_value = error_value + abs(test1[loc1-1][loc2-1] - test2[loc1-1][loc2-1])
        #print(abs(test1[loc1-1][loc2-1] - test2[loc1-1][loc2-1]))
    return error_value/99999

def data_split(filepath):
    ## the input is trainning dataset(for trainning model), test dataset(for providing location of predictions), n(size of neighbours)
    trainning_data = pd.read_csv(filepath, sep="\t", header = 0, names = ["userId", "movieId", "rating", "timeStamp"], index_col=False)
    testing_data = pd.pivot_table(trainning_data,values='rating',index='userId',columns='movieId')
    return trainning_data, testing_data

## run the knn algorithm
def knn_test(trainning_data, testing_data, n):

    ## for cross validation, the train data can keep the same with maybe different form
    data = knn(trainning_data, testing_data, n)
    data.loaddata()
    #data.get_similarity_with_user()
    data.get_similarity_with_movie()
    #data.find_n_neighbours()
    # sim_user_30_m = find_n_neighbours(data.similarity_with_movie,n)
    #test = knn(data.nmovie, data.nuser, data.Mean, data.final, data.final_movie, data.similarity_with_movie, sim_user_30_m, n)
    pred = data.pred()
    #print(data.check)

    return pred, data.check

def run():
    n = 30
    filepath = "ml-100k/u.data"
    trainning_data, testing_data = data_split(filepath)
    pred, test = knn_test(trainning_data, testing_data, n)
    print('RMSE for neighbour size' + str(n) + ' is ' + str(alternative_rmse(pred, test)))
    print('MAE for neighbour size' + str(n) + ' is ' + str(alternative_mae(pred, test)))

## for parameter tuning
def knn_para_tuning():

    filepath = "ml-100k/u.data"
    trainning_data, testing_data = data_split(filepath)

    test_start_time = time.time()

    for n in range(2, 31, 2):

        start_time = time.time()

        pred, test = knn_test(trainning_data, testing_data, n)

        elapsed_time = time.time() - start_time
        print("********************* knn for neighbour "+ str(i) + " Finished in : " + str(elapsed_time/60) + " minutes*********")

        ## test rmse and mae score for algorithms
        ## pred is the prediction result, data.check is the validation dataset
        ## for cross validation, maybe keep pred as the result of training result, the data_check for test data
        #print('RMSE with package sklearn.metrics for neighbour size' + str(i) + ' is '  + str(rmse(pred, data_check.fillna(0).values)))
        print('RMSE for neighbour size' + str(n) + ' is ' + str(alternative_rmse(pred, test)))
        #print('MAE with package sklearn.metrics for neighbour size' + str(i) + ' is ' + str(mae(pred, data_check.fillna(0).values)))
        print('MAE for neighbour size' + str(n) + ' is ' + str(alternative_mae(pred, test)))

    test_elapsed_time = time.time() - test_start_time
    print("total code finish in", test_elapsed_time, "seconds")

if __name__ == '__main__':
    # this one is to run the knn with n = 30
    run()
    # this one for parameter tuning
    #knn_para_tuning()


    ## todolist
    ## 1）knn 为 user based 而非 item based
    ##      1.1）根据 userimilarity matrix 找出对应user对应item的neighbours（必须已经看过该电影）
    ##      1.2）参考不同neighbour的weights给新给定item评分


########## 明天先看这个  ##############
    ## 2) knn 的输入， 看一下dataset和testset，输入similarity函数，输入k
    ## 3）不同similarity functions
    ## 4) baseline algorithm
    ## 5）cv函数编写

    ## 最后我这边可以提供三个算法，baseline，itembased knn, userbased knn
'''
## 我还需要
1） 写userbased knn
2） 尝试不同similarity function
3） 根据cv函数配适knn函数，使得knn只需要输入（train dataset，选择的 similarity function）

done  4)  rmse function 变为只需要输入ored 和 test dataset
这个稍等 5） 试其他evaluation function 如 recall
这个稍等 6） baseline algorithm

# 今晚：
# 写完class knn 的封装
明天：
写完similarity 函数+ userbased knn + 配适函数的输入
'''
