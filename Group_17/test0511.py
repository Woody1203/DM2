import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

class ubknn():

    def __init__(self, trainning_data, testing_data, n):
        # self.filepath = filepath
        self.n = n
        self.links_df = trainning_data
        ## locations of the predicted datapoints
        self.list_loc = list(testing_data.stack().index)
        self.nuser = 943
        self.nmovie = 1682

    def loaddata(self):
        ## load data
        ## mean values for each position according to each user
        self.Mean = self.links_df.groupby(by="userId",as_index=False)['rating'].mean()
        ## calculate how much the real rating is greater mean rating for each user
        self.links_df_avg = pd.merge(self.links_df,self.Mean,on='userId')
        self.links_df_avg['adg_rating']=self.links_df_avg['rating_x']-self.links_df_avg['rating_y']
        ## create pivot table for real taring
        self.check = pd.pivot_table(self.links_df_avg,values='rating_x',index='userId',columns='movieId')
        ## create pivot table for relative rating
        self.final = pd.pivot_table(self.links_df_avg,values='adg_rating',index='userId',columns='movieId')

        # Replacing NaN by Movie Average
        self.final_movie = self.final.fillna(self.final.mean(axis=0))

        # Replacing NaN by user Average
        self.final_user = self.final.apply(lambda row: row.fillna(row.mean()), axis=1)

    def get_similarity_with_user(self):

        sim = self.final_user.dot(self.final_user.T)
        norms = np.array([np.sqrt(np.diagonal(sim))])
        b=(sim / norms / norms.T)
        np.fill_diagonal(b.values, 0)
        self.similarity_with_user = pd.DataFrame(b,index=self.final_user.index)
        self.similarity_with_user.columns=self.final_user.index

    def get_similarity_with_movie(self):

        sim = self.final_movie.dot(self.final_movie.T)
        norms = np.array([np.sqrt(np.diagonal(sim))])
        b=(sim / norms / norms.T)
        np.fill_diagonal(b.values, 0)
        self.similarity_with_movie = pd.DataFrame(b,index=self.final_movie.index)
        self.similarity_with_movie.columns=self.final_movie.index

    def ub_user_item_score(self, user, item):
        ## users similary with the user
        a = self.sim_user_30_m[self.sim_user_30_m.index==user].values
        ## transform from dataframe to list
        b = a.squeeze().tolist()
        c = self.final_movie.loc[:,item]
        ## users who have seen the chosen item
        d = c[c.index.isin(b)]
        f = d[d.notnull()]
        avg_user = self.Mean.loc[self.Mean['userId'] == user,'rating'].values[0]
        index = f.index.values.squeeze().tolist()
        corr = self.similarity_with_user.loc[user,index]
        fin = pd.concat([f, corr], axis=1)
        fin.columns = ['adg_score','correlation']
        fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)
        nume = fin['score'].sum()
        deno = fin['correlation'].sum()
        final_score = avg_user + (nume/deno)
        return final_score

    def pred(self):
        self.pred = np.zeros((self.nuser,self.nmovie))
        self.sim_user_30_m = find_n_neighbours(self.similarity_with_user, self.n)
        ## predict every location the test dataset has ratings
        for loc1, loc2 in self.list_loc:

            try:
                self.score = self.ub_user_item_score(loc1, loc2)
                self.pred[loc1 - 1][loc2 - 1] = self.score
            except:
                self.pred[loc1 - 1][loc2 - 1] = 0

        return self.pred

def find_n_neighbours(df, n):
    order = np.argsort(df.values, axis=1)[:, :n]
    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
           .iloc[:n].index,
          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
    return df

## rmse calculate manual way
def mse(test1, test2):
    list_loc = test2.stack().index
    test2 = test2.fillna(0).values
    error_value = 0
    for loc1, loc2 in list_loc:
        error_value = error_value + ((test1[loc1-1][loc2-1] - test2[loc1-1][loc2-1])**2)
    error_value_mean = error_value/99999
    #root_mean_square_error = np.sqrt(error_value_mean)
    return error_value_mean

## mae calculate manual way
def mae(test1, test2):
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

def ub_knn_test(trainning_data, testing_data, n):

    testing_data = testing_data.replace(0, 'NaN')
    data = ubknn(trainning_data, testing_data, n)
    data.loaddata()
    data.get_similarity_with_user()
    pred = data.pred()
    #print(data.check)

    return pred, data.check

## for parameter tuning
def ub_knn_para_tuning():

    filepath = "ml-100k/u.data"
    trainning_data, testing_data = data_split(filepath)

    test_start_time = time.time()

    list_mse = []
    list_mae = []
    k = []

    for n in range(2, 31, 2):

        start_time = time.time()

        pred, test = ub_knn_test(trainning_data, testing_data, n)

        elapsed_time = time.time() - start_time
        print("********************************** knn for neighbour "+ str(n) + " Finished in : " + str(elapsed_time/60) + " minutes*********")

        ## test rmse and mae score for algorithms
        ## pred is the prediction result, data.check is the validation dataset
        mse_result = mse(pred, test)
        list_mse.append(mse_result)
        print('MSE for neighbour size' + str(n) + ' is ' + str(mse_result))
        mae_result = mae(pred, test)
        list_mae.append(mae_result)
        k.append(n)
        print('MAE for neighbour size' + str(n) + ' is ' + str(mae_result))

    test_elapsed_time = time.time() - test_start_time
    print("total code finish in", test_elapsed_time, "seconds")

    plt.plot(k, list_mse, "-b", label="MSE")
    plt.plot(k, list_mae, "-r", label="MAE")
    plt.legend(loc="upper right")
    #plt.ylim(0, 1.5)
    plt.title('MSE and MAE values according to diff k')
    plt.show()

def run_ubknn(n):
    filepath = "ml-100k/u.data"
    trainning_data, testing_data = data_split(filepath)
    print("this is trainning_data", trainning_data)
    df_new1, df_new2 = trainning_data[:200], trainning_data[200:]
    print("this is trainning_data after split", df_new2)

    print("this is testing_data", testing_data)
    ub_pred, test = ub_knn_test(df_new2, df_new1, n)
    print('MSE for neighbour size ' + str(n) + ' is ' + str(mse(ub_pred, df_new1)))
    print('MAE for neighbour size ' + str(n) + ' is ' + str(mae(ub_pred, df_new1)))

if __name__ == '__main__':

    ## knn parameter tuning
    # ub_knn_para_tuning()
    # run knn with specific k
    run_ubknn(10)
