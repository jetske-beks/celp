from data import UTILITY, SIMILARITY

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

def recommend(user_id='WxXB_DB_Im9mb00M8balAg', business_id='U_ihDw5JhfmSKBUUkpEQqw', city='agincourt', n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
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
    # get business data
    business = data.get_business(city, business_id)
    # get prediction
    stars = weighted_mean(select_neighborhood(user_id, business_id), user_id)
    # collect recommendation info
    prediction = {
        'business_id': business_id,
        'stars': stars,
        'name': business['name'],
        'city': city,
        'address': business['address']
    }
    
    return [prediction] * 10


if __name__ == '__main__':
    recommend()

