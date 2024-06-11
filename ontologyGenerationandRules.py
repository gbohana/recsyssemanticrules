# from google.colab import drive
# drive.mount('/content/drive')

# pip install Owlready2

# pip install rdflib

"""# 2 Creating Ontology

## Import Dependencies
"""

import pandas as pd
import numpy as np
from rdflib import Graph, URIRef, Literal, RDF, RDFS
import re
import math
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from os.path import exists

import os
os.environ['JAVA_OPTS'] = '-Xmx8g'

from owlready2 import *

#!java -Xmx8g /usr/local/lib/python3.10/dist-packages/owlready2/pellet/pellet-2.3.1

"""## Get Year From Regex"""

def get_release_year(movie_string):
    pattern = r"\((\d{4})\)"
    match = re.search(pattern, movie_string)

    if match:
        return int(match.group(1))
    else:
        return None

"""## Read Movies DB"""

movies_df = pd.read_csv('C:/Users/gabi_/Documents/_TCC/Code Python/movies.csv')
movies_df["year"] = get_release_year(str(movies_df.title))

movies_df.head()

"""## Read Ratings DB"""

ratings_df = pd.read_csv('C:/Users/gabi_/Documents/_TCC/Code Python/ratings.csv')

ratings_df.head()

movies_df = movies_df.sample(n=250, replace=False, random_state=42)
ratings_df = ratings_df[(ratings_df.movieId.isin(movies_df.movieId))]
print(ratings_df.size)
train_data, test_data = train_test_split(ratings_df, test_size=0.2, random_state=42)

"""## Populate Ontology"""

onto = get_ontology("http://test115.org/onto.owl")
with onto:
    # Define concepts
    class User(owl.Thing): pass
    class Evaluation(owl.Thing): pass
    class Movie(owl.Thing): pass
    class Genre(owl.Thing): pass

    class performsEvaluation(User >> Evaluation): pass
    class hasEvaluated(Evaluation >> Movie): pass
    class hasNotWatched(User >> Movie): pass
    class hasGenres(Movie >> Genre): pass
    class wasRecommended(User >> Movie): pass

    class hasRating(Evaluation >> float, FunctionalProperty): pass
    class hasYear(Movie >> int, FunctionalProperty): pass

# """### With Movies"""

# movies_onto = {}
# with onto:
#   for index, movie in movies_df.iterrows():
#       movie_uri = Movie("Movie_" + str(movie.movieId))
#       movie_uri.label = movie.title
#       movie_uri.hasYear = movie.year
#       movies_onto[movie.movieId] = movie_uri

#       for genre in movie.genres.split("|"):
#         genre_uri = Movie("Genre_" + genre)
#         movie_uri.hasGenres.append(genre_uri)

# """### With Evaluations"""

# users_onto = {}
# with onto:
#   for index, evaluation in train_data.iterrows():
#       if evaluation.userId not in users_onto:
#         user_uri = User('User_' + str(evaluation.userId))
#         users_onto[evaluation.userId] = user_uri
#       evaluation_uri = Evaluation('Evaluation_' + str(index))
#       users_onto[evaluation.userId].performsEvaluation.append(evaluation_uri)
#       evaluation_uri.hasEvaluated.append(movies_onto[evaluation.movieId])
#       evaluation_uri.hasRating = float(evaluation.rating)

# """### With users who not watched movies"""

# with onto:
#   userIds = test_data.userId.unique()
#   moviesIds = set(test_data.movieId.unique())

#   for userId in userIds:
#     if userId not in users_onto:
#         user_uri = User('User_' + str(userId))
#         users_onto[userId] = user_uri

#     watchedMovies = set(train_data[train_data.userId == userId].movieId)
#     moviesNotWatched = list(moviesIds - watchedMovies)

#     for movieId in moviesNotWatched:
#       users_onto[userId].hasNotWatched.append(movies_onto[movieId])

# """## Export Ontology"""

# onto_file = "/content/drive/MyDrive/MovieLens Python Project/movies_ontology.owl"
# onto.save(file=onto_file, format="rdfxml")

"""# 3 Generate Inferences"""

onto = get_ontology("movies_ontology.owl").load()

rule = "RULE_A"

with onto:
  if rule == "RULE_A":
    ruleA = Imp()
    ruleA.set_as_rule("performsEvaluation(?user, ?eval), hasEvaluated(?eval, ?movie), hasRating(?eval, 5.0), hasYear(?movie, ?year), hasYear(?movierec, ?year), hasNotWatched(?user, ?movierec) -> wasRecommended(?user, ?movierec)")
  elif rule == "RULE_B":
    ruleB = Imp()
    ruleB.set_as_rule("performsEvaluation(?user, ?eval), hasEvaluated(?eval, ?movie), hasRating(?eval, 5.0), hasGenres(?movie, ?genre), hasGenres(?movierec, ?genre), hasNotWatched(?user, ?movierec) -> wasRecommended(?user, ?movierec)")
  elif rule == "RULE_C":
    ruleC = Imp()
    ruleC.set_as_rule("performsEvaluation(?user, ?eval), hasEvaluated(?eval, ?movie), hasRating(?eval, ?r), greaterThan(?r, 3.0), hasGenres(?movie, ?genre), hasGenres(?movierec, ?genre), hasNotWatched(?user, ?movierec) -> wasRecommended(?user, ?movierec)")
  elif rule == "RULE_D":
    ruleD = Imp()
    ruleD.set_as_rule("performsEvaluation(?user, ?eval), hasEvaluated(?eval, ?movie), hasRating(?eval, ?r), greaterThan(?r, 3.0), hasYear(?movie, ?year), hasYear(?movierec, ?year), hasNotWatched(?user, ?movierec) -> wasRecommended(?user, ?movierec)")
  elif rule == "RULE_E":
    ruleE = Imp()
    ruleE.set_as_rule("performsEvaluation(?user, ?eval), hasEvaluated(?eval, ?movie), hasRating(?eval, 5.0), hasYear(?movie, ?year), hasYear(?movierec, ?year2), greaterThanOrEqual(?year, ?year2), hasNotWatched(?user, ?movierec) -> wasRecommended(?user, ?movierec)")
  elif rule == "RULE_F":
    ruleF = Imp()
    ruleF.set_as_rule("performsEvaluation(?user, ?eval), hasEvaluated(?eval, ?movie), hasRating(?eval, 5.0), hasYear(?movie, ?year), hasYear(?movierec, ?year2), greaterThanOrEqual(?year2, ?year), hasNotWatched(?user, ?movierec) -> wasRecommended(?user, ?movierec)")

  # ruleN = Imp()
  # ruleN.set_as_rule("performsEvaluation(?x, ?eval), hasEvaluated(?eval, ?m) -> hasYear(?m, 100000)")
  sync_reasoner_pellet(infer_property_values = True, infer_data_property_values = True)

"""## Generate CSV"""

def save_to_csv(csv_file, new_rows, col_names):
  if not os.path.isfile(csv_file):
    df = pd.DataFrame(columns=col_names)
    df.to_csv(csv_file, index=False)

  df = pd.read_csv(csv_file)

  df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

  df.to_csv(csv_file, index=False)

def save_recommendations(recommendations):
  recommendations_to_save = [];

  for index, recommendation in recommendations.iterrows():
    recommendations_to_save.append({"userId": recommendation.userId,
          "movieId": recommendation.movieId,
          "rating": recommendation.rating,
          "rule": recommendation.rule})

  tweetDF = pd.DataFrame(recommendations_to_save)

  save_to_csv("C:/Users/gabi_/Documents/_TCC/Code Python/recommendations.csv", tweetDF, ["userId", "movieId", "rating", "rule"])

"""## Get Inferences"""

inferences = []

with onto:
  for ind in onto.get_instances_of(User):
    if wasRecommended in ind.get_properties():
        for evaluation in ind.performsEvaluation:
            userId = int(float(ind.name.replace("User_","")))
            movieId = int(float(evaluation.hasEvaluated.first().name.replace("Movie_","")))
            inferences.append({"userId": userId, "movieId": movieId, "rating": evaluation.hasRating, "rule": rule})

generated_recommendations = pd.DataFrame(inferences)
generated_recommendations.head()
save_recommendations(generated_recommendations)

recommendations = pd.read_csv("C:/Users/gabi_/Documents/_TCC/Code Python/recommendations.csv")
len(recommendations)

"""# 4 Evaluating Results"""

def calculate_precision_recall_f1(y_true, y_pred):
    precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return precision, recall, f1_score

recommendations = pd.read_csv("C:/Users/gabi_/Documents/_TCC/Code Python/recommendations.csv")

generated_recommendations = pd.DataFrame(recommendations[recommendations.rule == rule])
generated_recommendations.head()

# Calculate Mean Absolute Error (MAE)
total_error = 0
n = len(ratings_df)

actual_ratings = []
predicted_ratings = []
binary_true = []
binary_pred = []
threshold = 4.0
generated_recommendations["rating"] = generated_recommendations.rating.astype(float)
for index, row in ratings_df.iterrows():
    userId = int(row["userId"])
    movieId = int(row["movieId"])
    actual_rating = row["rating"]
    predicted_rating = generated_recommendations.loc[(generated_recommendations['userId'] == userId) & (generated_recommendations['movieId'] == movieId), 'rating'].values
    if predicted_rating.size > 0:
      predicted_rating = predicted_rating[0]

      print(predicted_rating)
      binary_true.append(1 if actual_rating >= threshold else 0)
      binary_pred.append(1 if predicted_rating >= threshold else 0)

precision, recall, f1_score = calculate_precision_recall_f1(binary_true, binary_pred)
print("Precision:", precision)
print("Recall:", recall)
print("F1 score:", f1_score)
