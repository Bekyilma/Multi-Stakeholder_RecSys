from bokeh.core.json_encoder import pd
from . import get_logger, freeze, subdict
from sklearn import preprocessing
_LOG = get_logger('Recsys')


def process_user_info():
    HPs = pd.read_excel('/Users/bekyilma/Documents/Projects/vr/Rec/Alberto/Alberto.xlsx', index_col=0)
    HPs = HPs[['How likely are you interested in visiting popular paintings?',
               'How likely are you interested in visiting diverse content?',
               'How tolerant are you to crowd in exhibition areas?',
               'How tolerant are you towards walking in a museum?']]

    HPs = HPs.transpose()
    HPs.columns = ['Values']
    # Normalize Hyperparameters
    x = HPs[['Values']].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df_normalized = pd.DataFrame(x_scaled)

    HPs['norm_Values'] = df_normalized.values

    # Translate Hyperparameters

    Beta = HPs.at['How likely are you interested in visiting popular paintings?', 'norm_Values']
    Epsilon = HPs.at['How likely are you interested in visiting diverse content?', 'norm_Values']
    LAMBDA = HPs.at['How tolerant are you towards walking in a museum?', 'norm_Values']
    Crowd_tolerance = HPs.at['How tolerant are you to crowd in exhibition areas?', 'norm_Values']

    # -------------- Import preference information (weights) -------------- #
    df = pd.read_excel('/Users/bekyilma/Documents/Projects/vr/Rec/Alberto/Alberto.xlsx', index_col=0)
    df1 = df.drop(['Date submitted', 'Last page', 'Start language', 'Seed',
                   'Please select your choice below. Clicking on the "agree" button below indicates that: You have read the above information and you voluntarily agree to participate. If you do not wish to participate in the research study, please decline participation by clicking on the "disagree" button.',
                   'Which of the following best describes your museum visiting style?',
                   'How likely are you interested in visiting popular paintings?',
                   'How likely are you interested in visiting diverse content?',
                   'How tolerant are you to crowd in exhibition areas?',
                   'How tolerant are you towards walking in a museum?',
                   'Do you want to get contacted for a further interview to validate our recommender system and explain your choices?',
                   'If you answer YES, please provide your prefered means of communication? (email.)'], axis=1)
    df1.rename(columns={'000-0419-0000    \xa0': '000-0419-0000'}, inplace=True)

    selected_images_keys_list = list(df1.columns)
    # df1.dropna(axis=1, how='all')
    df1 = df1.transpose()
    df1.columns = ['a']
    df1.dropna(subset=['a'], inplace=True)
    df1['weights'] = df1['a'].astype('int')

    # Create x, where x the 'scores' column's values as floats
    x = df1[['weights']].values.astype(float)

    # Create a minimum and maximum processor object
    min_max_scaler = preprocessing.MinMaxScaler()

    # Create an object to transform the data to fit minmax processor
    x_scaled = min_max_scaler.fit_transform(x)

    # Run the normalizer on the dataframe
    df_normalized = pd.DataFrame(x_scaled)

    df1['norm_weights'] = df_normalized.values
    weights = df1['norm_weights'].tolist()

    # wrap preference info as a dictionary
    preference_dict = dict(zip(selected_images_keys_list, weights))

    _LOG.debug('Parameters {} {} {} {} {}'.format(Beta, Epsilon, LAMBDA, Crowd_tolerance, preference_dict))

    return Beta, Epsilon, LAMBDA, Crowd_tolerance, preference_dict
