import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

BASE_PATH = "C:/Users/MrLinh/Downloads/movie_reccommender/model"
PATH_TO_IMDB = f"{BASE_PATH}/movies_data/imdb_folder"
PATH_TO_RATING = f"{PATH_TO_IMDB}/ratings_small.csv"

PATH_TO_MLENS = f""

PATH_TO_FEATURE_STORE = f"{BASE_PATH}/feature_store_new.csv"
PATH_TO_VECTORIZED_TEXT = f"{BASE_PATH}/vectorized_text.pickle"

"""
ToDos:
1. filter out the top N based on text similarities with other movies
- dimensionality reduction of overview and soup
- plot of variance decomposition
- cosine similaritity
2. collaborative filtering: based on rating with other movies
- create rating matrix
- weighted mean
- reverse mapping between index and title


"""
FEATURE_STORE = pd.read_csv(PATH_TO_FEATURE_STORE)

COVARIATES = ["overview", "soup"]

X = FEATURE_STORE[COVARIATES]
def general_filtering():

# # vectorizer and add rating here?
#     def vectorizer(self): # reconsider
#         self.qualified_df["overview"] = self.qualified_df["overview"].fillna('').apply(lambda x: x[:316])
#         tfidf = TfidfVectorizer(stop_words="english")
#         self.tfidf_matrix = tfidf.fit_transform(self.qualified_df["overview"])
#         return self