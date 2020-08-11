import pandas as pd


def generate_stories():

                        # This module generate data related to paintings (curated stories) and encode data for Gurobi

    Recommendation_list = pd.read_csv('/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Data/Recommendations/LDA_recommendations.csv')

   # stories = pd.read_json('/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Data/datasets/stories.json')

                                #Story_groups: paintinglist

    Womens_Lives = ['000-04IB-0000', '000-04I6-0000', '000-04GY-0000', '000-02NQ-0000',
           '000-03NU-0000', '000-03P1-0000', '000-03EU-0000',
           '000-03OD-0000', '000-04H4-0000', '000-03ZZ-0000', '000-02UP-0000',
           '000-04HC-0000', '000-02OI-0000', '000-03Q9-0000', '000-03ZR-0000',
           '000-041C-0000', '000-04HM-0000', '000-02RL-0000', '000-04SJ-0000',
           '000-02QC-0000', '000-04I4-0000', '000-0437-0000', '000-02Q5-0000', '000-01AI-0000']

    Contemporary_Style_and_Fashion = ['000-04T5-0000', '000-03VT-0000', '000-04QC-0000',
           '000-02SW-0000', '000-01A9-0000', '000-03MW-0000',
           '000-044J-0000', '000-04LR-0000', '000-04PY-0000',
           '000-04K4-0000', '000-0400-0000', '000-04US-0000',
           '000-02VU-0000', '000-02W4-0000', '000-02U3-0000', '000-016Q-0000', '000-0307-0000', '000-01D1-0000',
           '000-03IY-0000', '000-042J-0000', '000-03U7-0000']

    Water = ['000-0344-0000','000-016V-0000', '000-05LR-0000', '000-03J4-0000','000-040J-0000','000-03TM-0000',
            '000-01AK-0000', '000-03QA-0000', '000-03ZI-0000', '000-04MC-0000',
           '000-04FG-0000', '000-03GX-0000', '000-04FU-0000', '000-042O-0000',
           '000-03TS-0000', '000-04I8-0000', '000-02ND-0000', '000-03GE-0000', '000-02VL-0000','000-04NP-0000']

    Women_Artists_and_Famous_Women = ['000-030D-0000','000-02VI-0000','000-02T4-0000', '000-03YZ-0000']

    Monsters_and_Demons = ['000-01D8-0000', '000-02NG-0000', '000-04JD-0000', '000-02U4-0000', '000-018G-0000',
                           '000-018Z-0000','000-01DK-0000', '000-02PC-0000', '000-02XP-0000', '000-03YQ-0000', '000-03LO-0000',
           '000-02SV-0000', '000-03M4-0000', '000-031A-0000', '000-02XU-0000',
           '000-02QB-0000', '000-04UI-0000', '000-02SL-0000', '000-032P-0000',
           '000-040L-0000', '000-03N1-0000', '000-02VD-0000', '000-03HY-0000',
           '000-043M-0000', '000-02UJ-0000', '000-04QP-0000']
    Migration_Journeys_and_Exile = ['000-03WL-0000', '000-01DN-0000', '000-019R-0000', '000-02UR-0000', '000-03G8-0000',
                                   '000-02U2-0000' ]

    Death = ['000-04PW-0000', '000-03TQ-0000', '000-02WO-0000','000-04J8-0000', '000-04S3-0000', '000-02ZT-0000']

    Battles_and_Commanders = ['000-03LN-0000', '000-02OU-0000', '000-03HA-0000', '000-04S4-0000', '000-03I6-0000']


    Warfare = ['000-0188-0000', '000-03V9-0000', '000-019X-0000', '000-017I-0000', '000-03GE-0000']


                        #Append stories to dataframe

    Recommendation_list.loc[Recommendation_list['Paintings'].isin(Womens_Lives), 'Stories'] = 'Womens Lives'
    Recommendation_list.loc[Recommendation_list['Paintings'].isin(Contemporary_Style_and_Fashion), 'Stories'] = 'Contemporary_Style_and_Fashion'
    Recommendation_list.loc[Recommendation_list['Paintings'].isin(Water), 'Stories'] = 'Water'
    Recommendation_list.loc[Recommendation_list['Paintings'].isin(Monsters_and_Demons), 'Stories'] = 'Monsters_and_Demons'
    Recommendation_list.loc[Recommendation_list['Paintings'].isin(Migration_Journeys_and_Exile), 'Stories'] = 'Migration_Journeys_and_Exile'
    Recommendation_list.loc[Recommendation_list['Paintings'].isin(Death), 'Stories'] = 'Death'
    Recommendation_list.loc[Recommendation_list['Paintings'].isin(Battles_and_Commanders), 'Stories'] = 'Battles_and_Commanders'
    Recommendation_list.loc[Recommendation_list['Paintings'].isin(Warfare), 'Stories'] = 'Warfare'
    #Recommendation_list.loc[Recommendation_list['Paintings'].isin(Womens_Lives), 'Stories'] = 'Womens Lives'
    Recommendation_list['Stories'] = Recommendation_list['Stories'].fillna('Uncategorised')

                    #label Uncategorized story groups

    Uncategorised = Recommendation_list[Recommendation_list['Stories'] == 'Uncategorised']
    Uncategorised = Uncategorised['Paintings'].tolist()

                        # Popularity

    """Additionally the openion of other people (crowd) on exhibits also has a potetial to influence 
        the preference of the user. Hence, we introduce a popularity score for. [Score (P,pop)]. 
        This score is extracted from The National Gallery website."""

                #Create apopular items list

    Must_see = ['000-0431-0000', '000-01AI-00000', '000-03JT-0000', '000-016U-0000', '000-01DF-0000', '000-0168-0000',
                '000-03G0-0000', '000-02X6-0000', '000-016Q-0000', '000-017R-0000', '000-02VI-0000', '000-019R-0000',
                '000-01AH-0000', '000-04PN-0000', '000-01CK-0000', '000-01B0-0000', '000-01AD-0000', '000-03TM-0000',
                '000-03HA-0000', '000-03TU-0000', '000-019M-0000', '000-043Q-0000', '000-02VY-0000', '000-019Z-0000',
                '000-01A0-0000', '000-01D6-0000', '000-0419-0000', '000-02T3-0000','000-01BA-0000', '000-031I-0000' ]

                #Add 'Score (P,pop) Column
    ind=[]
    for idx, value in Recommendation_list.iterrows():
        x = 0
        for u in Must_see:
            if u in value['Paintings']:
                ind.append(1)
                x = 1
                break
        if x == 0:
            ind.append(0)

    Recommendation_list['Score (P,pop)'] = ind


                        # Dump recommendation_df with stories and new scores
    Recommendation_list.to_csv('/Users/bekyilma/Documents/Projects/vr/Multi-Stakeholder_Recommendation/Data/Recommendations/Stories_+_LDA_recommendations.csv', index=False)

