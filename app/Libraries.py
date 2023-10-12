# # PROJECT KITCHEN

# # Author : Harini & Harshita
#
# ## Created Date  : September 26,2022
# ## Modified Date : October 05,2022

# # CONTENTS
#   This file has all the libraries used in the code required to run the model & data pre-processing

# # ALL REGIONS
#
# # 1. INPUT
# # 2. PRE-PROCESSING 
# #     2.1 NUMERIC
# #     2.2 CATEGORICAL
# #     2.3 TEXT
# # 3. K-MEDOIDS CLUSTERING WITH GOWER DISTANCE
# # 4. K-PROTOTYPE CLUSTERING (Not Using in the Output)
# # 5. DENDROGRAM
# # 6. OPTIMAL CLUSTERS
# # 6. CLUSTERING - WARD, SINGLE, COMPLETE, AVERAGE LINKAGES WITH ALL DISTANCE METRICS
# # 8. SIMILARITY SCORE
# # 9. HIGHEST SIMILARITY
# # 10. SIMILARITY CRUDE LOGIC
# # 11. MODEL INTERPRETABILITY
# # 12. OUTPUT 

# # Importing libraries and packages


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.feature_selection import VarianceThreshold
import math
from sklearn.preprocessing import MinMaxScaler
from itertools import product
import scipy.stats as ss

import nltk
nltk.data.path.append('/mnt/data/stopwords')
nltk.data.path.append('/mnt/data/wordnet')
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer,PorterStemmer
nltk.download('stopwords',download_dir='/mnt/data/stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet',download_dir='/mnt/data/wordnet')
import re
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer() 
from nltk.corpus import stopwords
stop = stopwords.words('english')
import string
string.punctuation
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import prince
from prince import MCA
import gower
# !pip install prince
# !pip install gower
#import mca

import scipy.cluster.hierarchy as shc
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
import matplotlib.cm as cm
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
#pip install shap
#pip install shapely
import shap

# Import module for k-protoype cluster
from kmodes.kprototypes import KPrototypes
import gower
from kmodes.kprototypes import KPrototypes
from sklearn_extra.cluster import KMedoids
from kneed import DataGenerator, KneeLocator
# !pip install kmodes
# !pip install kneed 
#kpro code
#pip install sklearn_extra
#pip install scikit-learn-extra
#pip install kmodes
#pip install kneed 


# # The End


