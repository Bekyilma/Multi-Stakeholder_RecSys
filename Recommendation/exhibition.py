import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import itertools
import re







# Path recommender
def path_recommender (Cr_t, LAMBDA, T_ava, Cr_s):

    #Cr_s is crowd size dictionary ... real time croed size per venue


    #Load Gurobi recomended POIs
    Recomended_POIs = pd.read_csv( '/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Results/MO_recommendations.csv')


    # project recommendation to venues
    ROOM_SCORE = Recomended_POIs.groupby(['Room_ID'])['Score_AG (P)', 'Time_est'].agg('sum')

    ROOM_SCORE = ROOM_SCORE.sort_values(by=['Score_AG (P)'], ascending=False)
    ROOM_SCORE['Room_Number'] = ROOM_SCORE.index
    ROOM_SCORE = ROOM_SCORE.rename({'Score_AG (P)': 'Score(R)'}, axis=1)


    #Import floor pan and compute distance between venues.

    Ng = pd.read_json(
        r'resources/datasets/ng-map-master-f57523e18f42d1565bace0a7f78effd00dc8f767/floors/floor-2/floor-2_rooms.geojson')
    json_normalize(Ng_map['features'])


    # We have Top x Pintings that are distributed in 17 rooms

    # The Path recommender needs to solve for N rooms:

    #(i) That give a total maximum score _Max_   $\sum_{K=1}^{N}$ Score (_R_$_{K}$)

    #(ii) The travel distance between consequtive rooms should be minimal _Min_   $\sum_{K=1}^{N}$ _dist_ (_R_$_{K}$, _R_$_{K+1}$)  Such that

    #The total estimated time for visiting and travel should not exceed the available time of the visitor. $\sum_{K=1}^{N}$ _T_$_{v}$(_R_$_{K}$)  + _T_(_R_$_{K}$, _R_$_{K+1}$) $\leq $ _T_$_{ava}$
