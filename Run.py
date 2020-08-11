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






    print(Recommendation_list)

   # _LOG.debug('Dataset = {LDA_recommendation}'.format(Recommendation_list))



def main():
    run()




if __name__ == '__main__':
    main()
