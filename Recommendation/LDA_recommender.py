import matplotlib.pyplot as plt
import cv2
#%matplotlib inline
import requests
from io import BytesIO
import urllib3
import os
import random
from imageio import imread
import pickle
import scipy.spatial
from IPython.core.display import HTML
import pandas as pd
import gensim
import numpy as np
import spacy
import re
import seaborn as sns
import random
from wordcloud import WordCloud
from gensim.models import ldamodel
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim
import os, re, operator, warnings
#warnings.filterwarnings('ignore')  # Let's not pay heed to them right now
#%matplotlib inline

import gurobi as gu
from PIL import Image, ImageOps
from keras.preprocessing import image
import matplotlib.image as mpimg
from urllib import request
import os
import random
from imageio import imread
import pickle
import scipy.spatial
from keras.preprocessing import image
from sklearn import preprocessing
from IPython.core.display import HTML
import sys
import pandas as pd
import gensim
import spacy
import re
import seaborn as sns
import random
from wordcloud import WordCloud
from gensim.models import ldamodel
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.wrappers.ldamallet import malletmodel2ldamodel
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
from gensim.models.wrappers import LdaMallet
from gensim.corpora import Dictionary
import pyLDAvis.gensim
from PIL import Image
#from collections.abc import Iterable
import collections

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


from . import get_logger


_LOG = get_logger('Recsys')


pd.options.display.max_colwidth = 50
path_to_model = '/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Data/models/lda.model'
path_to_cos_mat = '/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Data/matrices/lda/cosine-mat.npy'
path_to_topdoc_mat = '/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Data/matrices/lda/lda-output.npy'
painting_df = pd.read_csv('/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Data/datasets/ng-dataset.csv')
lda_model = LdaMallet.load(path_to_model)
lda = ldamodel.LdaModel.load(path_to_model)
print(lda)
lda.num_terms
cos_mat = np.load(path_to_cos_mat)
topdoc_mat = np.load(path_to_topdoc_mat)


# def topN_pid2Room(painting_df,top_n_pids):


def index2roomid(painting_df, index):
    try:
        room_id = painting_df.iloc[index].room
    except IndexError as ie:
        room_id = "Index '" + index + "' not found in dataset."
    return room_id


def indexlist2roomidlist(painting_df, index_list):
    roomid_list = [index2roomid(painting_df, index) for index in index_list]
    return roomid_list


def index2imageid(painting_df, index):
    try:
        image_id = painting_df.iloc[index].image
    except IndexError as ie:
        image_id = "Index '" + index + "' not found in dataset."
    return image_id


def indexlist2imageidlist(painting_df, index_list):
    imageid_list = [index2imageid(painting_df, index) for index in index_list]
    return imageid_list


def pid2index(painting_df, painting_id):
    """From the painting ID, returns the index of the painting in the painting dataframe
    Input:
            painting_df: dataframe of paintings
            painting_list: list of paintings ID (e.g ['000-02T4-0000', '000-03WC-0000...'])
    Output:
            index_list: list of the paintings indexes in the dataframe (e.g [32, 45, ...])
    """
    try:
        index = painting_df.loc[painting_df['painting_id'] == painting_id].index[0]
    except IndexError as ie:
        index = "Painting ID '" + painting_id + "' not found in dataset."
    return index


def pidlist2indexlist(painting_df, painting_list):
    """From a list of painting ID, returns the indexes of the paintings
    Input:
            painting_df: dataframe of paintings
            painting_list: list of paintings ID (e.g ['000-02T4-0000', '000-03WC-0000...'])
    Output:
            index_list: list of the paintings indexes in the dataframe (e.g [32, 45, ...])
    """
    index_list = [pid2index(painting_df, painting_id) for painting_id in painting_list]
    return index_list


def index2pid(painting_df, index):
    """From the index, returns the painting ID from the paintings dataframe
    Input:
            painting_df: dataframe of paintings
            index: index of the painting in the dataframe
    Output:
            pid: return the painting ID (e.g: 000-02T4-0000 )
    """
    try:
        pid = painting_df.iloc[index].painting_id
    except IndexError as ie:
        pid = "Index '" + index + "' not found in dataset."
    return pid


def indexlist2pidlist(painting_df, index_list):
    """From a list of indexes, returns the painting IDs
    Input:
            painting_df: dataframe of paintings
            index_list: list of the painting indexes in the dataframe
    Output:
            pid: list of paintings ID
    """
    pids_list = [index2pid(painting_df, index) for index in index_list]
    return pids_list


def recommend_paintings(painting_df, painting_list, weights, cos_mat, n):
    """Recommand paintings for a user based on a list of items that were liked
    Input:
            painting_df: dataframe of paintings
            painting_list: list of paintings index liked by a user
            cos_sim_mat: Cosine Similarity Matrix
            n: number of recommendation wanted
    Output:
            a list of indexes for recommended paintings
    """
    n_painting = len(painting_list)
    score_list = []
    index_list = pidlist2indexlist(painting_df, painting_list)

    for index in index_list:
        for w in weights:
            score = np.multiply(w, cos_mat[index])
            score[index] = 0
            score_list.append(score)
    score_list = np.sum(score_list, 0) / n_painting
    # score_list = np.sum(np.multiply(weights,score_list), axis=0) / N

    top_n_index = sorted(range(len(score_list)), key=lambda i: score_list[i], reverse=True)[:n]
    top_n_pids = indexlist2pidlist(painting_df, top_n_index)
    room_id = indexlist2roomidlist(painting_df, top_n_index)
    image_id = indexlist2imageidlist(painting_df, top_n_index)
    # print (type(top_n_pids))
    # print (type(score_list))
    val = np.sort(score_list)[::-1]
    # print(len(val))
    # print(len(score_list))
    dataset = pd.DataFrame(top_n_pids, columns=['Paintings'])
    dataset['Score (P,U)'] = val
    dataset['Room_ID'] = room_id
    dataset['image'] = image_id
    # dataset = pd.DataFrame({'Paintings': top_n_pids[:, 0], 'score(P,U)': val[:, 1]})
    # create a dataframe with top_n_pids(=>list) and their corresponding score(val => numpy array) of similarity

    #_LOG.debug('Dataset = {LDA_recommendation}'.format(dataset))

    return dataset












