import os
import json

def write_state(state, city, data_dir):
    """ Write the state to a new json file in the city dir. """
    with open(f"{data_dir}/{city}/state.json", 'w+') as f:
        f.write("{\"state\":\"%s\"}\n" % state)

def process_city(city, data_dir, state):
    """ Find the state a city is in. """
    with open(f"{data_dir}/{city}/business.json") as f:
        for line in f:
            business = json.loads(line)
            if 'state' in business:
                write_state(business['state'], city, data_dir)
                return (business['state'] == state)
    return False

def process_cities(data_dir, state=None):
    """ Process all cities to find their states. """
    city_list = []
    cities = os.listdir(data_dir)
    for city in cities:
        if process_city(city, data_dir, state):
            city_list.append(city)
    # return a list with cities in state 'state'
    return city_list


if __name__ == '__main__':
    data_dir = 'data'
    process_cities(data_dir)
