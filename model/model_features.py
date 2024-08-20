import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle 

from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer

# 'C:\Users\MrLinh\Downloads\movie_reccommender\model'
# ENV = os.getenv("MOVIE_REC","movie_recommender")
BASE_PATH = "C:/Users/MrLinh/Downloads/movie_reccommender/model"
# BASE_PATH = os.getenv("","/movies_data") # fill in gcp server info
# GS_BUCKET = os.getenv("","") # fill in gcp bucket info

PATH_TO_IMDB = f"{BASE_PATH}\movies_data\imdb_folder"
PATH_TO_IMDB_MOVIE = f"{PATH_TO_IMDB}\movies_metadata.csv"
PATH_TO_KEYWORDS = f"{PATH_TO_IMDB}\keywords.csv"
PATH_TO_RATING = f"{PATH_TO_IMDB}\ratings_small.csv"

PATH_TO_MLENS = f""

PATH_TO_FEATURE_STORE = f"{BASE_PATH}\feature_store.csv"
PATH_TO_VECTORIZED_TEXT = f"{BASE_PATH}\vectorized_text.pickle"

def data_path(movie_db): # reconsider
    if movie_db == "imdb":
        return PATH_TO_IMDB
    else:
        return PATH_TO_MLENS
    

def read_data(movie_db) -> pd.DataFrame:
    data_path = data_path(movie_db) # call on both datasets
    data = pd.read_csv(data_path) 
    return data

class FeatureBuilder:
    def __init__(self, raw_df: pd.DataFrame, lowtime = 45, hightime = 300):
        self.raw_df = raw_df[(raw_df["runtime"] >= lowtime) & (raw_df["runtime"] <= hightime)]
        self.qualified_df = pd.DataFrame() # not necessary?
        self.tfidf_matrix = None

    def weighted_scores(self, percentile = 0.8): 
        minimum_vote = self.raw_df["vote_count"].quantile(percentile)
        qualified_movies = self.raw_sdf[self.raw_df["vote_count" >= minimum_vote]]
        average_vote = qualified_movies["vote_average"].mean()
        qualified_movies["score"] = ((qualified_movies["vote_count"]/(qualified_movies["vote_count"]+minimum_vote))*qualified_movies["vote_average"]) + ((minimum_vote/(qualified_movies["vote_count"] + minimum_vote))*average_vote)
        # sanity check by checking scores distribution min = 0 max = 10
        plt.hist(qualified_movies["scores"], bins = 100)
        plt.grid(True)
        plt.savefig("score_distribution.png")

        self.qualified_df = qualified_movies
        return self

    def destring(self):
        self.qualified_df["genres"] = self.qualified["genres"].fillna([]).apply(literal_eval)
        self.qualified_df["genres_list"] = self.qualified_df["genres"].apply(lambda x: [i["name"] for i in x] if isinstance(x,list) else [])
        self.qualified_df = self.qualified_df.explode("genres_list").drop("genres",axis=1)
        return self
    
    def vectorizer(self): # be careful when call this
        self.qualified_df["overview"] = self.qualified_df["overview"].fillna('')
        tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = tfidf.fit_transform(self.qualified_df["overview"])
        return self

def run_processing_pipeline(
        feature_builder: FeatureBuilder
) -> pd.DataFrame:
    return (
        feature_builder.weighted_scores()
        .destring()
        .vectorizer()
    )

print("done")
# if __name__ == "__main__":

#     for db in ["imdb"]:
#         movie_data = read_data(db)
#         intermediate = run_processing_pipeline(movie_data)
#         feature_store = intermediate.qualified_df
#         tfdif_matrix = intermediate.tfidf_matrix
#         feature_store.to_csv(PATH_TO_FEATURE_STORE) # f string here to differentiate the 2
#         with open(PATH_TO_VECTORIZED_TEXT, "wb") as file: # here also
#             pickle.dump(tfdif_matrix, file)

