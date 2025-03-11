import os
import sys
import pickle
import warnings
import matplotlib.pyplot as plt
import numpy as np

##################
# PATHS
path_allen='../data/AllenMouseConnectome/'
path_fig='../figures/Figure3/'
path_empirical_data='../data/Empirical_data_features/'

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_pickle(data, filename):
    with open(filename, "wb") as file:
        return pickle.dump(data, file)

def extract_and_concatenate_upper_triangular(array):
    num_matrices = array.shape[0]
    matrix_size = array.shape[1]
    upper_triangular_vectors = [array[i][np.triu_indices(matrix_size, k=1)] for i in range(num_matrices)]
    concatenated_vector = np.concatenate(upper_triangular_vectors)
    return concatenated_vector

warnings.filterwarnings("ignore")

LABESSIZE = 10
plt.rcParams["axes.labelsize"] = LABESSIZE
plt.rcParams["xtick.labelsize"] = LABESSIZE
plt.rcParams["ytick.labelsize"] = LABESSIZE


# BOLD_file = (
#     "../data/SBI/BOLD_data.pkl"
# )
# groups = ["CTRL", "RSC", "ACA", "Th"]

# BOLD = load_pickle(BOLD_file)

# n_subjects = {}
# for group in groups:
#     for subj in BOLD[group].keys():
#         n_subjects[group] = len(BOLD[group].keys()) // 2
# n_subjects

# NDP = {}
# FCs = {}
# dFCs = {}
# for group in groups:
#     NDP[group] = {}
#     FCs[group] = {}
#     dFCs[group] = {}
#     for subj in BOLD[group].keys():
#         z=stats.zscore(BOLD[group][subj][-925:,:],axis=0)
#         NDP[group][subj] = np.mean(z,axis=1)**2        
#         FCs[group][subj] = np.corrcoef(z.T)
#         dFCs[group][subj] = np.corrcoef(go_edge(z))

# save_pickle(NDP,path_empirical_data+'Empirical_NDP.pkl')
# save_pickle(FCs,path_empirical_data+'Empirical_FunctionalConnectivity.pkl')
# save_pickle(dFCs,path_empirical_data+'Empirical_DynamicFunctionalConnectivity.pkl')