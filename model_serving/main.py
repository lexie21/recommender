from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import pickle
import functions_framework
from google.cloud import storage
from ast import literal_eval
import random

BUCKET_NAME = ""
SIMILARITY_MATRIX = ()
COVARIATES = ["title","release_date","runtime"]

BASE_PATH = "C:/Users/MrLinh/Downloads/movie_reccommender/model"
PATH_TO_IMDB = f"{BASE_PATH}/movies_data/imdb_folder"

PATH_TO_FEATURE_STORE = f"{BASE_PATH}/feature_store_new.csv"
PATH_TO_COSINE_MATRIX = "C:/Users/MrLinh/Downloads/movielens/ml-100k/cosine_matrix.pickle"

app = Flask(__name__)

FEATURE_STORE = pd.read_csv(PATH_TO_FEATURE_STORE)
FEATURE_STORE["release_date"] = FEATURE_STORE["release_date"].apply(lambda x: int(x.split('-')[0]) if isinstance(x, str) else 0)
FEATURE_STORE["genres"] = FEATURE_STORE["genres"].apply(literal_eval)
exploded_df = FEATURE_STORE.explode("genres")

def _get_similarity_matrix():
    with open(PATH_TO_COSINE_MATRIX, "rb") as file:
        simil_matrix = pickle.load(file)
    simil_df = pd.DataFrame(simil_matrix,index=FEATURE_STORE["id"],columns=FEATURE_STORE["id"])
    return simil_df

@app.route("/",methods=["GET"])
def index():
    idx = random.sample(range(FEATURE_STORE.shape[0]),4)
    cached_df = FEATURE_STORE.iloc[idx][["id","title","poster_path"]]
    """Modify this when call external db"""
    ids = cached_df["id"]
    titles = cached_df["title"]
    urls = cached_df["poster_path"]
    tuples = zip(ids, titles, urls)
    return render_template("index.html", tuples = tuples)

@app.route("/recommend",methods=["POST"])
def recommend(): 
    movie_ids = [int(id) for id in request.form.getlist('movies_id')]
    def genres_extract():
        genres_list = []
        for id in movie_ids:
            genres_list += (exploded_df[exploded_df["id"] == id]["genres"].to_list())
        filtered_df = exploded_df[exploded_df["genres"].apply(lambda x: x in set(genres_list))].drop_duplicates(subset="id")
        return filtered_df.sort_values(by="score",ascending=False)[1:11]["id"]
    def simil_extract(): 
        simil_id = []
        simil_df = _get_similarity_matrix()
        for id in movie_ids:
            simil_id += simil_df[id].sort_values(ascending=False).index.to_list()[1:6]
        simil_df = exploded_df.drop_duplicates(subset="id")
        simil_id =  simil_df[simil_df["id"].apply(lambda x: x in simil_id)].sort_values(by="score",ascending=False)["id"]
        return simil_id
       
    id1 = genres_extract().to_list()
    id2 = simil_extract().to_list()
    id_final = id1 + id2
    df = FEATURE_STORE[FEATURE_STORE["id"].apply(lambda x: x in id_final)][COVARIATES].sample(frac=1)
    df_json = df.to_json(orient='records')
    return jsonify(df_json)

if __name__ == "__main__":
    app.run(debug=True)


