import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ast import literal_eval

"""
Resolve git
"""
# 'C:\Users\MrLinh\Downloads\movie_reccommender\model'
# ENV = os.getenv("MOVIE_REC","movie_recommender")
BASE_PATH = "C:/Users/MrLinh/Downloads/movie_reccommender/model"
# BASE_PATH = os.getenv("","/movies_data") # fill in gcp server info
# GS_BUCKET = os.getenv("","") # fill in gcp bucket info

PATH_TO_IMDB = f"{BASE_PATH}/movies_data/imdb_folder"
PATH_TO_IMDB_MOVIE = f"{PATH_TO_IMDB}/movies_metadata.csv"
PATH_TO_KEYWORDS = f"{PATH_TO_IMDB}/keywords.csv"

PATH_TO_FEATURE_STORE = f"{BASE_PATH}/feature_store_new.csv"

def read_data(data_path) -> pd.DataFrame:
    data = pd.read_csv(data_path)
    return data

class FeatureBuilder:
    def __init__(self, raw_df: pd.DataFrame, raw_keywords: pd.DataFrame, lowtime = 45, hightime = 300):
        # self.raw_df = raw_df[(raw_df["runtime"] >= lowtime) & (raw_df["runtime"] <= hightime)]
        self.raw_df = raw_df
        self.raw_df.drop(self.raw_df[self.raw_df["id"].apply(lambda x: '-' in x)].index, inplace=True)
        self.raw_df["id"] = self.raw_df["id"].astype(int)
        self.raw_df["overview"].fillna('',inplace=True)
        self.raw_df["overview"] = self.raw_df["overview"].apply(lambda x: str.lower(x[:316]) if (len(x) > 316) else str.lower(x))
        self.raw_df = pd.merge(self.raw_df, raw_keywords, on="id")
        self.qualified_df = pd.DataFrame() 

    def weighted_scores(self, percentile = 0.8): 
        minimum_vote = self.raw_df["vote_count"].quantile(percentile)
        qualified_movies = self.raw_df[self.raw_df["vote_count"] >= minimum_vote]
        average_vote = qualified_movies["vote_average"].mean()
        qualified_movies["score"] = ((qualified_movies["vote_count"]/(qualified_movies["vote_count"]+minimum_vote))*qualified_movies["vote_average"]) + ((minimum_vote/(qualified_movies["vote_count"] + minimum_vote))*average_vote)
        
        plt.hist(qualified_movies["score"], bins = 100)
        plt.grid(True)
        plt.savefig("score_distribution.png")

        self.qualified_df = qualified_movies
        return self

    def destring(self, vars):
        def generate_list(x):
            if isinstance(x, list):
                names = [i["name"] for i in x]
                if len(names) > 3:
                    names = names[:3]
                return names
            return []

        # sanitize data to prevent ambiguity, remove space and convert to lowercase
        def sanitize(x):
            if isinstance(x, list):
                return [str.lower(i.replace(" ", "")) for i in x] 
            else:
                if isinstance(x, str):
                    return str.lower(x.replace(" ", ""))
                else:
                    return ''
        
        def create_soup(x):
            return ' '.join(x["keywords"]) + ' '.join(x["genres"])
                
        for var in vars:
            self.qualified_df[var] = self.qualified_df[var].fillna('[]').apply(literal_eval)
            self.qualified_df[var] = self.qualified_df[var].apply(generate_list)
            self.qualified_df[var] = self.qualified_df[var].apply(sanitize)
        
        self.qualified_df["soup"] = self.qualified_df.apply(create_soup, axis=1) 
       
        return self

def run_processing_pipeline(
        feature_builder: FeatureBuilder
) -> pd.DataFrame:
    return (
        feature_builder.weighted_scores()
        .destring(vars=["keywords", "genres"])
    )

if __name__ == "__main__":
    movie_data = read_data(PATH_TO_IMDB_MOVIE)
    keyword_data = read_data(PATH_TO_KEYWORDS)
    intermediate = run_processing_pipeline(FeatureBuilder(movie_data, keyword_data))

    feature_store = intermediate.qualified_df
    feature_store.to_csv(PATH_TO_FEATURE_STORE, index=False)
