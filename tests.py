import data
import recommender
import pandas as pd
import numpy as np
import argparse
import time

def split_train_test(user_ratings, test_size=25):
  user_ratings_train = pd.Series()
  user_ratings_test = pd.Series()
  for user in user_ratings.index:
    rand_index = np.random.choice(user_ratings[user].index, size=test_size, replace=False)
    train_mask = [i not in rand_index for i in user_ratings[user].index]
    test_mask = [i in rand_index for i in user_ratings[user].index]
    user_ratings_train[user] = user_ratings[user][train_mask]
    user_ratings_test[user] = user_ratings[user][test_mask]
  return user_ratings_train, user_ratings_test

def build_utility_matrix(user_ratings):
  #df = pd.DataFrame(columns=user_ratings.index)
  lst = []
  for user in user_ratings.index:
    lst.append(user_ratings[user])
  df = pd.concat(lst, axis=1, sort=False)
    #for business in user_ratings[user].index:
      #df[user][business] = user_ratings[user][business]

  return df

def top_5_test_collab(user_ratings_test, utility, similarity):
  predicted_ratings = pd.Series()
  for user in user_ratings_test.index:
    prediction = pd.Series()
    businesses = user_ratings_test[user].index
    for business in businesses:
      prediction.at[business] = recommender.weighted_mean(recommender.select_neighborhood(user, business), user) 
    predicted_ratings[user] = prediction.astype(object)
    predicted_ratings[user].sort_values(inplace=True)

  known_top_10 = pd.Series()
  for user in user_ratings_test.index:
    known_top_10[user] = user_ratings_test[user].sort_values()[:10]

  predicted_5 = pd.Series()
  for user in predicted_ratings.index:
    predicted_5[user] = predicted_ratings[user][:5]

  correct = 0
  for user in predicted_5.index:
    correct += sum([1 for b in predicted_5[user].index if b in known_top_10[user].index])
  precission = correct / (predicted_5.size * 5)
  print('precission collaborative: {}'.format(precission))

  mse = 0
  for user in predicted_ratings.index:
    mse += sum([abs(predicted_ratings[user][b] - user_ratings_test[user][b]) for b in predicted_ratings[user].index])

  mse /= (predicted_ratings.size * 25)
  print('MSE collaborative: {}'.format(mse))

def top_5_test_content(user_ratings_test, utility, similarity):
  predicted_ratings = pd.Series()
  for user in user_ratings_test.index:
    businesses = user_ratings_test[user].index
    predicted_ratings[user] = recommender.content_prediction(user, businesses, utility, similarity).astype(object)
    predicted_ratings[user].sort_values(inplace=True)

  known_top_10 = pd.Series()
  for user in user_ratings_test.index:
    known_top_10[user] = user_ratings_test[user].sort_values()[:10]

  predicted_5 = pd.Series()
  for user in predicted_ratings.index:
    predicted_5[user] = predicted_ratings[user][:5]

  correct = 0
  for user in predicted_5.index:
    correct += sum([1 for b in predicted_5[user].index if b in known_top_10[user].index])
  precission = correct / (predicted_5.size * 5)
  print('precission content: {}'.format(precission))

  mse = 0
  for user in predicted_ratings.index:
    mse += sum([abs(predicted_ratings[user][b] - user_ratings_test[user][b]) for b in predicted_ratings[user].index])

  mse /= (predicted_ratings.size * 25)
  print('MSE content: {}'.format(mse))

def top_5_test_hybrid(user_ratings_test, utility, sim, sim_content):
  predicted_ratings_collab = pd.Series()
  for user in user_ratings_test.index:
    prediction = pd.Series()
    businesses = user_ratings_test[user].index
    for business in businesses:
      prediction.at[business] = recommender.weighted_mean(recommender.select_neighborhood(user, business), user) 
    predicted_ratings_collab[user] = prediction.astype(object)
    predicted_ratings_collab[user].sort_values(inplace=True)

  predicted_ratings_content = pd.Series()
  for user in user_ratings_test.index:
    businesses = user_ratings_test[user].index
    predicted_ratings_content[user] = recommender.content_prediction(user, businesses, utility, sim_content).astype(object)
    predicted_ratings_content[user].sort_values(inplace=True)

  known_top_10 = pd.Series()
  for user in user_ratings_test.index:
    known_top_10[user] = user_ratings_test[user].sort_values()[:10]

  predicted_ratings_hybrid = pd.Series()
  for user in user_ratings_test.index:
    predicted_ratings_hybrid[user] = 0.78 * predicted_ratings_collab[user] + 0.22 * predicted_ratings_content[user]

  predicted_5 = pd.Series()
  for user in predicted_ratings_hybrid.index:
    predicted_5[user] = predicted_ratings_hybrid[user][:5]

  correct = 0
  for user in predicted_5.index:
    correct += sum([1 for b in predicted_5[user].index if b in known_top_10[user].index])
  precission = correct / (predicted_5.size * 5)
  print('precission hybrid: {}'.format(precission))

  mse = 0
  for user in predicted_ratings_hybrid.index:
    mse += sum([abs(predicted_ratings_hybrid[user][b] - user_ratings_test[user][b]) for b in predicted_ratings_hybrid[user].index])

  mse /= (predicted_ratings_hybrid.size * 25)
  print('MSE hybrid: {}'.format(mse))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--collab', action='store_true')
  parser.add_argument('--content', action='store_true')
  parser.add_argument('--hybrid', action='store_true')
  parser.add_argument('--state', default='OH')
  parser.add_argument('-n', default=-1)
  options = parser.parse_args()

  start = time.time()

  data.initialisation(state=options.state, n=int(options.n), test=True)
  user_ratings = pd.Series()
  for column in data.UTILITY:
    ur = data.UTILITY[column].dropna()
    if ur.size > 50:
      user_ratings[column] = ur.astype(object)

  print(user_ratings.size)
  if user_ratings.size == 0:
    exit(0)
  user_ratings_train, user_ratings_test = split_train_test(user_ratings)
  new_utility = build_utility_matrix(user_ratings_train)

  similarity = None
  if options.collab:
    similarity = data.calculate_similarity(new_utility)
    top_5_test_collab(user_ratings_test, new_utility, similarity)
  
  if options.content:
    top_5_test_content(user_ratings_test, new_utility, data.SIMILARITY_CATEGORIES)

  if options.hybrid: 
    if not options.collab:
      similarity = data.calculate_similarity(new_utility)
    top_5_test_hybrid(user_ratings_test, new_utility, similarity, data.SIMILARITY_CATEGORIES)
    

  end = time.time()
  print('Tests took {}s'.format(end - start))
