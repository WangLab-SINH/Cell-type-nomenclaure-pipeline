import argparse
import subprocess
import math
import matplotlib as mpl
import warnings
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import os
from sklearn.tree import export_text
import re
import shutil
import collections
import operator

import create_dataset_perl
import Random_forest_algo_database
import rename_file_perl
import read_nb_result
import random

# 忽略一些版本不兼容等警告
warnings.filterwarnings("ignore")
np.random.seed(1)
random.seed(1)

total_info_file = pd.read_csv("/picb/neurosys/chiyuhao/0429/qiricheng/code/info_qrc_all_0612.csv", encoding='gbk')
total_file_path = "/picb/neurosys/chiyuhao/0429/qiricheng/data/"

for current_index in range(total_info_file.shape[0]):
    current_line = total_info_file.iloc[current_index]
    print(current_line)
    if current_line['files'] == 'T1':
        current_file_path = total_file_path + str(current_line['pmid']) + '/' + current_line['description'] + '/'
    else:
        current_file_path = total_file_path + str(current_line['pmid']) + '/'

    if os.path.exists(current_file_path + "umap.rds"):
        if os.path.exists(current_file_path + "cluster_table_new3.csv"):
            if os.path.exists(current_file_path + "cluster_table_new4.csv"):
                continue

            create_dataset_training_data_name = current_file_path + "new_data.csv"
            create_dataset_training_meta_name = current_file_path + "new_anno.csv"
            cluster_table_name = current_file_path + "cluster_table_new3.csv"

            #########################################random forest

            result_dir = current_file_path + "result1/"
            output_train_file = current_file_path + "NB_input_new1/"
            output_test_file = current_file_path + "NB_input_new1/"
            auc_output_file = current_file_path + "accuracy.txt"

            if not os.path.exists(result_dir):
                os.mkdir(result_dir)
            if not os.path.exists(output_train_file):
                os.mkdir(output_train_file)
            if not os.path.exists(output_test_file):
                os.mkdir(output_test_file)


            cluster_table = pd.read_csv(cluster_table_name)
            # if 'cluster_new' not in cluster_table.columns:
            #     cluster_table.loc[:, 'cluster_new'] = cluster_table['cluster']
            # cluster_table['de_flag'] = "False"
            process_type = []
            for i in range(cluster_table.shape[0]):
                temp = cluster_table.iloc[i]
                if pd.isnull(cluster_table.iloc[i]['gene1']):
                    process_type.append(cluster_table.iloc[i]['cluster_new'])
            if len(process_type) < 1:
                cluster_table.to_csv((current_file_path + "cluster_table_new4.csv"), index=False)
                continue

            meta_data = pd.read_csv(create_dataset_training_meta_name, index_col=0)
            if len(set(meta_data['cluster_label'])) == 1:
                cluster_table.to_csv((current_file_path + "cluster_table_new4.csv"), index=False)
                continue
            train_data = pd.read_csv(create_dataset_training_data_name, index_col=0)
            train_data = train_data.T
            train_data['label'] = 'B'
            all_type = set(meta_data['cluster_label'])
            process_type = list(set(process_type) & all_type)
            for type in process_type:
                print("NA111111111111111111111111111111111111111")
                print(type)
                current_train = train_data.copy()
                current_train['label'][list(meta_data['cluster_label'] == type)] = 'A'
                cell_type_name = type
                model_file = result_dir + type.replace("/", "") + ".txt"
                model_clas_path_file = result_dir + type.replace("/", "") + "_class_path.txt"
                # if os.path.exists(model_clas_path_file):
                #     continue
                new_output_train_file = output_train_file + type.replace("/", "") + ".txt"
                new_output_test_file = output_test_file + type.replace("/", "") + ".txt"
                Random_forest_algo_database.random_forest_main(current_train, current_train, new_output_train_file,
                                                               new_output_test_file, model_file, model_clas_path_file,
                                                               auc_output_file, cell_type_name)

                new_anno = pd.read_csv(result_dir + type.replace("/", "") + ".csv")

                node_file = pd.read_table(
                    result_dir + type.replace("/", "") + "_class_path.txt",
                    sep=" ", header=None)
                node_file = node_file[node_file[8] == "R"]
                current_candidate_gene = set(node_file[3])
                new_anno1 = new_anno[np.in1d(np.array(new_anno["1"]), np.array(list(current_candidate_gene)))]
                cluster_table['gene1'] = cluster_table.gene1.astype(str)
                cluster_table['gene2'] = cluster_table.gene2.astype(str)
                cluster_table['gene3'] = cluster_table.gene3.astype(str)
                cluster_table['de_flag'] = cluster_table.de_flag.astype(str)
                if new_anno1.shape[0] >= 3:
                    print("here1")
                    gene1 = new_anno1.iloc[0, 2]
                    gene2 = new_anno1.iloc[1, 2]
                    gene3 = new_anno1.iloc[2, 2]
                    cluster_table.at[cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'gene1'] = gene1
                    cluster_table.at[cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'gene2'] = gene2
                    cluster_table.at[cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'gene3'] = gene3
                    cluster_table.at[
                        cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'de_flag'] = 'TREE'
                elif new_anno1.shape[0] >= 2:
                    print("here2")
                    gene1 = new_anno1.iloc[0, 2]
                    gene2 = new_anno1.iloc[1, 2]
                    cluster_table.at[cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'gene1'] = gene1
                    cluster_table.at[cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'gene2'] = gene2
                    cluster_table.at[
                        cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'de_flag'] = 'TREE'
                elif new_anno1.shape[0] >= 1:
                    print("here3")
                    gene1 = new_anno1.iloc[0, 2]
                    cluster_table.at[cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'gene1'] = gene1
                    cluster_table.at[
                        cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'de_flag'] = 'TREE'
                else:
                    if new_anno.shape[0] >= 3:
                        gene1 = new_anno.iloc[0, 2]
                        gene2 = new_anno.iloc[1, 2]
                        gene3 = new_anno.iloc[2, 2]
                        cluster_table.at[
                            cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'gene1'] = gene1
                        cluster_table.at[
                            cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'gene2'] = gene2
                        cluster_table.at[
                            cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'gene3'] = gene3
                        cluster_table.at[
                            cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'de_flag'] = 'TREE'
                    elif new_anno.shape[0] >= 2:
                        gene1 = new_anno.iloc[0, 2]
                        gene2 = new_anno.iloc[1, 2]
                        cluster_table.at[
                            cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'gene1'] = gene1
                        cluster_table.at[
                            cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'gene2'] = gene2
                        cluster_table.at[
                            cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'de_flag'] = 'TREE'
                    elif new_anno.shape[0] >= 1:
                        gene1 = new_anno.iloc[0, 2]
                        cluster_table.at[
                            cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'gene1'] = gene1
                        cluster_table.at[
                            cluster_table[cluster_table['cluster_new'] == type].index.tolist()[0], 'de_flag'] = 'TREE'
            cluster_table.to_csv((current_file_path + "cluster_table_new4.csv"), index=False)




