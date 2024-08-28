from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import pickle
import functions_framework
from google.cloud import storage
import request

BUCKET_NAME = ""
SIMILARITY_MATRIX = ()
COVARIATES = ["title","release_date","runtime","poster_path"]

def _get_images():
    pass

def _get_similarity_matrix():
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(BUCKET_NAME)
    blob_model = bucket.blob(SIMILARITY_MATRIX)
    with blob_model.open("rb") as handle:
        simil_matrix = pickle.load(handle)
    return simil_matrix

@functions_framework.http
def index(request):
    images = _get_images()
    # set <images> in index.html
    return render_template("index.html")

@functions_framework.http
# given movie_ids, return the recommendations
def recommend(request):
    pass


