from data import UTILITY, SIMILARITY, UTILITY_CATEGORIES, SIMILARITY_CATEGORIES, CITIES
from operator import itemgetter

import data
import numpy
import pandas
import random
import numpy

def select_neighborhood(user, business):
    """ Selects all items with similarity > 0. """
    # error handling
    if user not in UTILITY or business not in SIMILARITY:
        return pandas.Series()
    # get all businesses that the target user has rated
    rated = UTILITY[user].dropna().index
    # get all similarity scores of those businesses with the target
    scores = SIMILARITY[business].loc[rated]
    # drop those lower than 0
    return scores[scores > 0]

def weighted_mean(neighborhood, user):
    # error handling
    if neighborhood.empty:
        return 0
    # get the user ratings for the neighborhood
    ratings = UTILITY[user].loc[neighborhood.index]
    # calculate the predicted rating
    return (ratings * neighborhood).sum() / neighborhood.sum()

def mse(predicted):
    # get the difference
    difference = predicted['stars'] - predicted['prediction']
    # return the mean square error
    return numpy.square(difference).sum() / len(predicted)

def baseline():
    data = pandas.DataFrame(index=UTILITY.index, columns=UTILITY.columns)
    avg = UTILITY.apply(numpy.mean)

def baseline_prediction(data):
    # get all unique ids
    business_ids = data['business_id'].unique()
    user_ids = data['user_id'].unique()
    # add a 'predicted rating' column
    data['prediction'] = pandas.Series(numpy.nan, index=data.index)
    print(" * Starting predict test for %i businesses..." % len(business_ids))
    # predict a rating for every business
    count = 0
    for business in business_ids:
        count += 1
        print("   %i" % count)
        for user in user_ids:
            # calculate neighborhood & get prediction
            prediction = data.loc[data['business_id'] == business, 'stars'].mean()
            # add to the data
            data.loc[(data['business_id'] == business) & (data['user_id'] == user), 'prediction'] = prediction
    return data

def predict(data):
    """ Predict the ratings for all items in the data. """
    # get all unique ids
    business_ids = data['business_id'].unique()
    user_ids = data['user_id'].unique()
    # add a 'predicted rating' column
    data['prediction'] = pandas.Series(numpy.nan, index=data.index)
    print(" * Starting predict test for %i businesses..." % len(business_ids))
    # predict a rating for every business
    count = 0
    for business in business_ids:
        count += 1
        print("   %i" % count)
        for user in user_ids:
            # calculate neighborhood & get prediction
            prediction = weighted_mean(select_neighborhood(user, business), user)
            # add to the data
            data.loc[(data['business_id'] == business) & (data['user_id'] == user), 'prediction'] = prediction
    return data

def content_prediction(user_id, business_ids, utility, similarity):
    """
    Make prediction for all businesses based on content similarity
    """
    ratings = utility[user_id].dropna()
    predictions = pandas.Series()
    counter = 0
    for business in business_ids:
        if not business in similarity:
            counter += 1
            continue
        sim = similarity[business]
        predictions.at[business] = (ratings * sim.loc[ratings.index]).mean()
    print(len(business_ids))
    print(counter)
    return predictions

def recommend_collab(businesses, user, business_id=None):
    prediction_list = []
    for business in businesses:
        if business['business_id'] == business_id:
            continue
        # save info about the business
        prediction = {
            'id': business['business_id'],
            'count': business['review_count'],
            'avg': business['stars'],
            'city': business['city'].lower()
        } 
        # get prediction
        prediction['rating'] = weighted_mean(select_neighborhood(user['user_id'], prediction['id']), user['user_id'])
        prediction_list.append(prediction)

    sorted_list = sorted(prediction_list, key=itemgetter('city', 'rating', 'count'))

    return sorted_list

def recommend_content(businesses, user, business_id=None):
    business_ids = [b['business_id'] for b in businesses]
    predictions = content_prediction(user['user_id'], business_ids, UTILITY, SIMILARITY_CATEGORIES)
    predictions.sort_values(inplace=True);
    
    return predictions

    
def recommend(user=None, business_id=None, city=None, n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user, business_id and/or city.
    A recommendation is a dictionary in the form of:
        {
            business_id:str
            stars:str
            name:str
            city:str
            adress:str
        }
    """
    global UTILITY, SIMILARITY, UTILITY_CATEGORIES, SIMILARITY_CATEGORIES, CITIES

    # read in matrices
    if not UTILITY:
        UTILITY = pandas.read_pickle('utility.pkl')
    if not SIMILARITY:
        data.SIMILARITY = pandas.read_pickle('similarity.pkl')

    if not UTILITY_CATEGORIES:
        UTILITY_CATEGORIES = pandas.read_pickle('utility_content.pkl')
    if not SIMILARITY_CATEGORIES:
        SIMILARITY_CATEGORIES = pandas.read_pickle('similarity_content.pkl')

    # fill in user, city
    if not user:
        user = data.get_city_users('eastlake')[0]
    if not city:
        city = 'eastlake'

    state = get_state(city)
    cities = load_cities(state)

    # get state businesses for recommendations
    businesses = data.load(cities, 'business')

    # make predictions for all businesses in the cities
    predictions_collab = recommend_collab(businesses, user, business_id)
    predictions_content = recommend_content(businesses, user)

    if predictions_content.empty:
        raise Exception("Predictions are empty!")
    recommend_list = []
    for p in predictions_content[:n].index:
        business = [b for b in businesses if b['business_id'] == p][0]
        recommend_list.append({
            'business_id': p,
            'stars': business['stars'],
            'name': business['name'],
            'city': business['city'],
            'address': business['address']
        })

    return (recommend_list * n)[:n]


if __name__ == '__main__':
    recommend()

