import os
import ruamel.yaml as yaml
import pandas as pd
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import ward, fcluster, dendrogram
from activities.gdocs_open import get_spreadsheet_data
from activities.clustering import eqsc
import matplotlib.pyplot as plt

with open("config/config.yml") as f:
    config = yaml.load(f, Loader=yaml.Loader)


def format_spreadsheet_data(data_list):
    data_responses = data_list[1:]
    data_header = data_list[0]
    data_df = pd.DataFrame(data_responses)
    data_df.columns = data_header
    data_df.dropna(how="all", inplace=True)
    data_df.reset_index(inplace=True, drop=True)
    return data_df


def get_answers_similarity_matrix(data_df, answers_loc):
    data_answers_only_np = data_df.iloc[:, answers_loc:].to_numpy()
    return squareform(pdist(data_answers_only_np, "euclidean"))


def save_dendrogram(data_df, config, team=None):
    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, "out")
    similarity_matrix_answers = get_answers_similarity_matrix(
        data_df, config["starting_question"]
    )
    Z = ward(squareform(similarity_matrix_answers))
    clusters_unbalanced = fcluster(Z, t=3, criterion="maxclust")
    data_df["clusters_unbalanced"] = clusters_unbalanced
    data_df[["Email address", "Name", "clusters_unbalanced"]].to_excel(
        os.path.join(out_dir, f"clusters_{team}.xlsx"), index=False
    )

    name_column_name = config["name_column_name"]
    labels = list(data_df[name_column_name])
    plt.figure(figsize=(20, 25))
    d = dendrogram(
        Z,
        labels=labels,
        orientation="right",
        distance_sort="descending",
        leaf_font_size=20,
    )

    plt.savefig(
        os.path.join(out_dir, f"dendrogram_{team}_plt.png"),
        format="png",
        bbox_inches="tight",
    )


def main():
    data = get_spreadsheet_data(config["spreadsheet_id"], config["sheet_name"])
    data_df = format_spreadsheet_data(data)
    team_list = list(data_df["What is your Merkle team unit?"].unique())

    save_dendrogram(data_df, config, "full")
    for team in team_list:
        data_team = data_df[data_df["What is your Merkle team unit?"] == team]
        save_dendrogram(data_team, config, team)


if __name__ == "__main__":
    main()
