from data import UTILITY, SIMILARITY
from operator import itemgetter

import data
import pandas
import random

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

def recommend(user=None, business_id='U_ihDw5JhfmSKBUUkpEQqw', city=None, n=10):
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
    # read in matrices
    if not UTILITY:
        data.UTILITY = pandas.read_pickle('utility.pkl')
    if not SIMILARITY:
        data.SIMILARITY = pandas.read_pickle('similarity.pkl')
    # fill in user, city, business
    if not user:
        user = data.get_user('John')
    if not city:
        city = user['city']
    # get city data
    businesses = data.get_city(city)
    while len(businesses) < n:
        city = random.choice(data.load_cities())
        business = data.get_city(city)
        businesses.extend(business)

    # make predictions for all businesses in the city
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

    recommend_list = []
    for prediction in sorted_list[:n]:
        # get business data
        business = data.get_business(prediction['city'], prediction['id'])
        # collect recommendation info
        recommendation = {
            'business_id': prediction['id'],
            'stars': prediction['avg'],
            'name': business['name'],
            'city': prediction['city'],
            'address': business['address']
        }
        recommend_list.append(recommendation)
    
    return recommend_list


if __name__ == '__main__':
    recommend()

