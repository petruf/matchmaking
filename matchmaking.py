# -*- coding: utf-8 -*-

import yaml
import pandas as pd
import numpy as np
import spacy
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import ward, fcluster
import itertools
from activities.gdocs_open import get_spreadsheet_data
from activities.clustering import eqsc


with open('config/config.yml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

nlp = spacy.load('en_core_web_md')


def get_text_distance(text1_np, text2_np):
    text1_spacy = nlp(text1_np[0])
    text2_spacy = nlp(text2_np[0])
    return text1_spacy.similarity(text2_spacy)


def format_spreadsheet_data(data_list):
    data_responses = data_list[1:]
    data_header = data_list[0]
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


if __name__ == '__main__':
    data = get_spreadsheet_data(config['spreadsheet_id'], config['sheet_name'])
    data_df = format_spreadsheet_data(data)
    resp_no = len(data_df)
    square_const = int(resp_no * (resp_no - 1) /2)
    
    similarity_matrix_answers = get_answers_similarity_matrix(data_df, config['starting_question'])
    similarity_matrix_profile = np.zeros((resp_no, resp_no))
    for i in config['nlp_analyzed_columns']:
        similarity_matrix_profile_columns = get_profile_similarity_matrix(data_df, i)
        similarity_matrix_profile -= similarity_matrix_profile_columns
    
    similarity_matrix = similarity_matrix_answers + similarity_matrix_profile
    similarity_matrix_final = assign_limit_weights(data_df, np.copy(similarity_matrix), config['gender_column_name'])
    
    # optimization of matched pairs
    similarity_matrix_final_noise = similarity_matrix_final + squareform(np.random.normal(loc=0, scale=0.001, size=square_const))
    row_ind, col_ind = linear_sum_assignment(similarity_matrix_final_noise)
    data_df['Match'] = data_df[config['name_column_name']].iloc[col_ind].reset_index(drop=True)
    data_df['Match_id'] = col_ind
    
    # clustering of pre-finalized similarity matrix
    Z = ward(squareform(similarity_matrix_answers))
    clusters_unbalanced = fcluster(Z, t=2, criterion='maxclust')
    clusters_balanced = eqsc(similarity_matrix, K=2)
    data_df['clusters_unbalanced'] = clusters_unbalanced
    data_df['clusters_balanced'] = clusters_balanced
    
    # saving to excel spreadsheet
    writer = pd.ExcelWriter('matchmaking.xlsx')
    data_df.to_excel(writer, sheet_name='matchmaking_results')
    pd.DataFrame(similarity_matrix_answers).to_excel(writer, sheet_name='answers_matrix')
    pd.DataFrame(similarity_matrix_profile).to_excel(writer, sheet_name='profile_matrix')
    pd.DataFrame(similarity_matrix).to_excel(writer, sheet_name='unadjusted_matrix')
    pd.DataFrame(similarity_matrix_final).to_excel(writer, sheet_name='final_matrix')
    writer.save()
    