import sys
import os
import logging
import numpy as np
import Recommendation
import pandas as pd
from sklearn.utils import check_random_state
from textwrap import dedent
from sklearn import preprocessing


_LOG = Recommendation.get_logger('Recsys')


def run():
    painting_df = pd.read_csv('/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Data/datasets/ng-dataset.csv')
    path_to_cos_mat = '/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Data/matrices/lda/cosine-mat.npy'
    cos_mat = np.load(path_to_cos_mat)
    Beta, Epsilon, LAMBDA, Crowd_tolerance, preference_dict = Recommendation.user.process_user_info()
    Rated_painting_list = []
    WEAIGHTS = []

    for k, v in preference_dict.items():
        Rated_painting_list.append(k)
        WEAIGHTS.append(v)


    Recommendation_list = Recommendation.LDA_recommender.recommend_paintings(painting_df, Rated_painting_list, WEAIGHTS, cos_mat, 2367)

    # Normalize Scores
    x = Recommendation_list[['Score (P,U)']].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled)

    Recommendation_list['Score (P,U)'] = df_normalized.values

    #Dump_LDA recommendations_df

    Recommendation_list.to_csv('/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Data/Recommendations/LDA_recommendations.csv', index=False)


    Recommendation.painting.generate_stories()

    #Retrive Recommendation_list_df with stories

    Recommendation_list = pd.read_csv('/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Data/Recommendations/Stories_+_LDA_recommendations.csv')

    Recommendation_list['Score_AG (P)'] = Recommendation_list['Score (P,U)'] + Recommendation_list['Score (P,pop)'].apply(lambda x: x * Beta)


            # Normalize Score_AG (P)

    nm = Recommendation_list[['Score_AG (P)']].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    nm_scaled = min_max_scaler.fit_transform(nm)
    df_normalized = pd.DataFrame(nm_scaled)

    Recommendation_list['Score_AG (P)'] = df_normalized.values


    #Estimate visiting time based on Visiting style (User provided)

    Time_estimator = pd.read_excel('/Users/bekyilma/Documents/Projects/vr/Rec/Alberto/Alberto.xlsx', index_col=0)
    #Time_estimator.set_index('Response ID')
    #Visitng_style = Time_estimator[['Which of the following best describes your museum visiting style?']]
    Visitng_style = Time_estimator.at[ 40 , 'Which of the following best describes your museum visiting style?']

    print("THIS IS VISITING STYLE", Visitng_style)

    if Visitng_style == 'I frequently change the direction of my tour, usually avoiding empty space. I see almost all exhibits, but times vary between exhibits.':
        #The Butterfly visitor:
        Recommendation_list['Time_est'] = np.random.randint(15, 60, size=len(Recommendation_list))
        
    elif Visitng_style == 'I walk mostly through empty space making just a few stops and see most of the exhibits but for a short time.':
        #The Fish visitor: 
        Recommendation_list['Time_est'] = np.random.randint(15, 60, size=len(Recommendation_list))

    elif Visitng_style == 'I see only exhibits I am  interested in. I walk through empty space and stays for a long time only in front of selected exhibits.':
        # The Grasshopper visitor:
        Recommendation_list['Time_est'] = np.random.randint(15, 60, size=len(Recommendation_list))

    elif Visitng_style == 'I spend a long time observing all exhibits and moves close to the walls and the exhibits avoiding empty space.':
        # The Ant visitor:
        Recommendation_list['Time_est'] = np.random.randint(15, 60, size=len(Recommendation_list))



                 #Baseline Recommender

    Policy_I = Recommendation_list.sort_values('Score_AG (P)', ascending=False)

                #Dump Baseline recommendation

    Policy_I.to_csv('/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Data/Recommendations/Policy_I_recommendations.csv', index= False)


    print(Recommendation_list)

   # _LOG.debug('Dataset = {LDA_recommendation}'.format(Recommendation_list))



def main():
    run()




if __name__ == '__main__':
    main()
