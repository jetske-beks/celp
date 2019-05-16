"""
This file loads the data from the data directory and shows you how.
Feel free to change the contents of this file!
Do ensure these functions remain functional:
    - get_business(city, business_id)
    - get_reviews(city, business_id=None, user_id=None, n=10)
    - get_user(username)
"""

import os
import json
import time
import numpy
import pandas
import random

DATA_DIR = "yelp-all"

CITIES = {}
BUSINESSES = {}
REVIEWS = {}
USERS = {}

UTILITY = []
SIMILARITY = []

# - - - - - - - - - - - - - - - - load functions - - - - - - - - - - - - - - - #

def load_cities():
    """
    Finds all cities (all directory names) in ./DATA_DIR
    Returns a list of city names
    """
    return os.listdir(DATA_DIR)


def load(cities, data_filename, to_remove=[]):
    """
    Given a list of city names,
        for each city extract all data from ./DATA_DIR/<city>/<data_filename>.json
    Returns a dictionary of the form:
        {
            <city1>: [<entry1>, <entry2>, ...],
            <city2>: [<entry1>, <entry2>, ...],
            ...
        }
    """
    data = {}
    for city in cities:
        city_data = []
        with open(f"{DATA_DIR}/{city}/{data_filename}.json", "r") as f:
            for line in f:
                l = json.loads(line)
                for key in to_remove:
                    l.pop(key, None)
                city_data.append(l)
        data[city] = city_data
    return data

# - - - - - - - - - - - - - - - helper functions - - - - - - - - - - - - - - - #

def check(business, treshold):
    return business['review_count'] >= treshold

def trim(reviews, to_delete):
    return [r for r in reviews if not r['business_id'] in to_delete]

def mem_usage(panda):
    if isinstance(panda, pandas.DataFrame):
        usage_b = panda.memory_usage(deep=True).sum()
    else:
        # we assume if not a df it's a series
        usage_b = panda.memory_usage(deep=True)
    # convert bytes to megabytes
    usage_mb = usage_b / 1024 ** 2
    return "{:03.2f} MB".format(usage_mb)

def to_pandas(CITIES, SET):
    set_list = []
    for city in CITIES:
        set_frame = pandas.DataFrame.from_dict(SET[city]).assign(city=city)
        set_list.append(set_frame)
    # append all datapoints to one DataFrame
    return pandas.concat(set_list, ignore_index=True, sort=True)

def optimize(frame, types):
    # convert columns to optimal types
    frame = frame.astype(types)
    # optimize float and int sizes
    floats = frame.select_dtypes(include=['float'])
    converted_float = floats.apply(pandas.to_numeric, downcast='float')
    frame[converted_float.columns] = converted_float
    ints = frame.select_dtypes(include=['int'])
    converted_int = ints.apply(pandas.to_numeric, downcast='unsigned')
    frame[converted_int.columns] = converted_int
    # return finished DataFrame
    return frame

def cosine_similarity(business1, business2):
    """ Calculate cosine similarity between two businesses. """
    # select for users that have rated both businesses
    index1 = numpy.argwhere(~numpy.isnan(business1))
    index2 = numpy.argwhere(~numpy.isnan(business2))
    selected = numpy.intersect1d(index1, index2)
    if not selected.any():
        return 0
    # get the ratings
    ratings1 = business1[selected]
    ratings2 = business2[selected]
    # calculate cosine similarity
    numerator = (ratings1 * ratings2).sum()
    denumerator = numpy.sqrt((ratings1 ** 2).sum()) * numpy.sqrt((ratings2 ** 2).sum())
    return numerator / denumerator if denumerator != 0 else 0

def calculate_similarity(utility):
    """ Creates similarity matrix based on cosine similarity. """
    from scipy.spatial.distance import pdist, squareform
    matrix = squareform(pdist(utility, cosine_similarity))
    numpy.fill_diagonal(matrix, 1)
    return pandas.DataFrame(matrix, columns=utility.index, index=utility.index)

# - - - - - - - - - - - - - - - initialisation - - - - - - - - - - - - - - - - #

def initialisation(n=-1):
    global CITIES, BUSINESSES, REVIEWS
    global UTILITY, SIMILARITY

    print(" * Loading data for %i cities..." % n)

    # load data
    CITIES = load_cities()[:n]
    BUSINESSES = load(CITIES, "business")
    REVIEWS = load(CITIES, "review", ['funny', 'cool', 'useful', 'text', 'date'])

    # remove all businesses with fewer than <treshold> reviews
    treshold = 10
    review_count = sum([len(REVIEWS[city]) for city in CITIES])
    print(" * Number of businesses: %i" % sum([len(BUSINESSES[city]) for city in CITIES]), end="")
    for city in CITIES:
        businesses = []
        to_delete = []
        for business in BUSINESSES[city]:
            if check(business, treshold):
                businesses.append(business)
            else:
                to_delete.append(business['business_id'])
        BUSINESSES[city][:] = businesses
        REVIEWS[city][:] = trim(REVIEWS[city], to_delete)
    print(" → %i" % sum([len(BUSINESSES[city]) for city in CITIES]))
    print(" * Number of reviews: %i → %i" % (review_count, sum([len(REVIEWS[city]) for city in CITIES])))

    # convert to pandas DataFrame
    business = to_pandas(CITIES, BUSINESSES)
    reviews = to_pandas(CITIES, REVIEWS)
    # optimize memory usage
    print(" * Memory usage (businesses): %s" % mem_usage(business), end="")
    business = optimize(business, {'is_open': 'bool', 'review_count': 'int', 'state': 'category', 'city': 'category'})
    print(" → %s" % mem_usage(business))
    print(" * Memory usage (reviews): %s" % mem_usage(reviews), end="")
    reviews = optimize(reviews, {'city': 'category'})
    print(" → %s" % mem_usage(reviews))
    
    # calculate utility matrix
    start = time.time()
    utility = reviews.pivot_table(index='business_id', columns='user_id', values='stars')
    end = time.time()
    print(" * Calculating utility matrix took %f seconds" % (end - start))

    # mean-center the matrix
    utility = utility - utility.mean()

    utility.to_pickle('utility.pkl')
    UTILITY = utility

    # calculate similarity matrix
    start = time.time()
    similarity = calculate_similarity(utility)
    end = time.time()
    print(" * Calculating similarity matrix took %f seconds" % (end - start))

    similarity.to_pickle('similarity.pkl')
    SIMILARITY = similarity

# - - - - - - - - - - - - - functions used in app.py - - - - - - - - - - - - - #

def get_business(city, business_id):
    """
    Given a city name and a business id, return that business's data.
    Returns a dictionary of the form:
        {
            name:str,
            business_id:str,
            stars:str,
            ...
        }
    """
    with open(f"{DATA_DIR}/{city}/business.json", "r") as f:
        for line in f:
            business = json.loads(line)
            if business["business_id"] == business_id:
                return business

    raise IndexError(f"invalid business_id {business_id}")


def get_reviews(city, business_id=None, user_id=None, n=10):
    """
    Given a city name and optionally a business id and/or a user id,
    return n reviews for that business/user combo in that city.
    Returns a dictionary of the form:
        {
            text:str,
            stars:str,
            ...
        }
    """
    def should_keep(review):
        if business_id and review["business_id"] != business_id:
            return False
        if user_id and review["user_id"] != user_id:
            return False
        return True

    reviews = []
    with open(f"{DATA_DIR}/{city}/review.json", "r") as f:
        for line in f:
            review = json.loads(line)
            if should_keep(review):
                reviews.append(review)

    return random.sample(reviews, min(n, len(reviews)))


def get_user(username):
    """
    Get a user by its username
    Returns a dictionary of the form:
        {
            user_id:str,
            name:str,
            ...
        }
    """
    for city in load_cities():
        with open(f"{DATA_DIR}/{city}/user.json", "r") as f:
            for line in f:
                user = json.loads(line)
                if user['name'] == username:
                    user['city'] = city
                    return user

    raise IndexError(f"invalid username {username}")

# - - - - - - - - - - - - - - more getter functions - - - - - - - - - - - - - - #

def get_usercity(user_id):
    """
    Get the city of a user by its user_id
    Returns a city in the form of a string
    """
    for city in load_cities():
        with open(f"{DATA_DIR}/{city}/user.json", "r") as f:
            for line in f:
                user = json.loads(line)
                if user['user_id'] == user_id:
                    return city

    raise IndexError(f"invalid username {username}")

def get_city(city):
    """
    Given a city name, return the data for all businesses.
    Returns an array of the form:
        [<business1>, <business2>, ...]
    """
    with open(f"{DATA_DIR}/{city}/business.json", "r") as f:
        business_list = []
        for line in f:
            business = json.loads(line)
            business_list.append(business)

    return business_list


if __name__ == '__main__':
    initialisation(n=10)
