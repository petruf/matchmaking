# -*- coding: utf-8 -*-

import yaml
import pandas as pd
import numpy as np
import spacy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
import itertools
from sklearn.neighbors import DistanceMetric
from activities.gdocs_open import get_spreadsheet_data

with open('config/config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

nlp = spacy.load('en_core_web_md')
    
def get_text_distance(text1_np, text2_np):
    text1_spacy = nlp(text1_np[0])
    text2_spacy = nlp(text2_np[0])
    return text1_spacy.similarity(text2_spacy)


def format_spreadsheet_data(data_list):
    data_responses = data[1:]
    data_header = data[0]
    data_df = pd.DataFrame(data_responses)
    data_df.columns = data_header
    data_df.dropna(how='all', inplace=True)
    data_df.reset_index(inplace=True, drop=True)
    return data_df


def get_profile_similarity_matrix(data_df, profile_column_name):
    texts_np = data_df[profile_column_name].to_numpy().reshape(-1,1)
    return squareform(pdist(texts_np, get_text_distance))


def get_answers_similarity_matrix(data_df, answers_loc):
    data_answers_only_np = data_df.iloc[:, answers_loc:].to_numpy()
    return squareform(pdist(data_answers_only_np, 'euclidean'))


def assign_limit_weights(data_df, matrix, gender_column_name):
    genders = data_df[gender_column_name].to_dict()
    for i in itertools.combinations(genders, 2):
        if genders[i[0]] == genders[i[1]]:
            matrix[i[0], i[1]] = 99999
            matrix[i[1], i[0]] = 99999
    np.fill_diagonal(matrix, 9999999)
    return matrix


data = get_spreadsheet_data(config['spreadsheet_id'], config['sheet_name'])
data_df = format_spreadsheet_data(data)
similarity_matrix_answers = get_answers_similarity_matrix(data_df, 6)
similarity_matrix_profile = get_profile_similarity_matrix(data_df, 'V angličtině krátce popište svého ideálního partnera (3 věty)')
similarity_matrix = similarity_matrix_answers - similarity_matrix_profile
similarity_matrix_final = assign_limit_weights(data_df, similarity_matrix, 'Pohlaví')

row_ind, col_ind = linear_sum_assignment(similarity_matrix)
data_df['Match'] = data_df['Jméno'].iloc[col_ind].reset_index(drop=True)
