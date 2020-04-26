import pandas as pd

def pearson(user1, user2, df):
    '''
    Calculates similarity between two users. Takes user's ids and dataframe as inputs.
    '''

    df_short = df[df[user1].notnull() & df[user2].notnull()]

    if len(df_short) == 0:
        return 0

    else:
        rat1 = [row[user1] for i, row in df_short.iterrows()]
        rat2 = [row[user2] for i, row in df_short.iterrows()]

        numerator = sum([(rat1[i] - sum(rat1)/len(rat1)) * (rat2[i] - sum(rat2)/len(rat2)) for i in range(0,len(df_short))])
        denominator1 = sum([(rat1[i] - sum(rat1)/len(rat1)) ** 2 for i in range(0,len(df_short))])
        denominator2 = sum([(rat2[i] - sum(rat2)/len(rat2)) ** 2 for i in range(0,len(df_short))])

        if denominator1 * denominator2 == 0:
            return 0
        else:
            return numerator / ((denominator1 * denominator2) ** 0.5)

def get_neighbours(user_id, df):
    '''
    Creates a sorted list of users, who are most similar to specified user. Calculate similarity between current user and
    all other users and sort by similarity.
    '''
    distances = [(user, pearson(user_id, user, df)) for user in df.columns if user != user_id]

    distances.sort(key=lambda x: x[1], reverse=True)

    distances = [i for i in distances if i[1] > 0]
    return distances

def recommend(user, df, n_users=2, n_recommendations=2):
    '''
    Generate recommendations for the user. Take userID and Dataframe as input. Get neighbours and get weighted score for
    each place they rated. Return sorted list of places and their scores.
    '''

    recommendations = {}
    nearest = get_neighbours(user, df)

    n_users = n_users if n_users <= len(nearest) else len(nearest)

    user_ratings = df[df[user].notnull()][user]

    place_ratings = []

    for i in range(n_users):
        neighbour_ratings = df[df[nearest[i][0]].notnull()][nearest[i][0]]
        for place in neighbour_ratings.index:
            if place not in user_ratings.index:
                place_ratings.append([place,neighbour_ratings[place],nearest[i][1]])

    recommendations = get_ratings(place_ratings)
    return recommendations[:n_recommendations]

def get_ratings(place_ratings):

    '''
    Creates Dataframe from list of lists. Calculates weighted rarings for each place.
    '''

    ratings_df = pd.DataFrame(place_ratings, columns=['placeID', 'rating', 'weight'])

    ratings_df['total_weight'] = ratings_df['weight'].groupby(ratings_df['placeID']).transform('sum')
    recommendations = []

    for i in ratings_df.placeID.unique():
        place_ratings = 0
        df_short = ratings_df.loc[ratings_df.placeID == i]
        for j, row in df_short.iterrows():
            place_ratings += row[1] * row[2] / row[3]
        recommendations.append((i, place_ratings))

    recommendations = [i for i in recommendations if i[1] >= 1]

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

links_df = pd.read_csv("ml-100k/u.data", sep="\t", header = 0, names = ["userId", "movieId", "rating", "timeStamp"], index_col=False)
links_df['rating'] = links_df['rating'].apply(lambda x: 0.000001 if x == 0 else x)
movie_ratings_df=links_df.pivot_table(index='movieId',columns='userId',values='rating')
movie_ratings_df.head()

results = recommend(401, movie_ratings_df,10,20)

print(results)
