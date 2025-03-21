import os
import sys
import torch
import pickle
import warnings
import numpy as np
import pandas as pd
from os.path import join
from copy import deepcopy
import scipy.stats as stats
from numpy import linalg as LA
from scipy.stats import zscore
from typing import Union, List
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.decomposition import PCA
from vbi.models.cupy.mpr import MPR_sde
from vbi import get_features_by_domain, get_features_by_given_names, report_cfg, update_cfg
from vbi import extract_features, extract_features_df

warnings.filterwarnings("ignore")

LABESSIZE = 10
plt.rcParams["axes.labelsize"] = LABESSIZE
plt.rcParams["xtick.labelsize"] = LABESSIZE
plt.rcParams["ytick.labelsize"] = LABESSIZE


class G_Dataset:

    nregions = 148
    remove_roi = [
        27,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        101,
        129,
        130,
        131,
        132,
        133,
        134,
        135,
        136,
        137,
        138,
        139,
        140,
        141,
        142,
        143,
        144,
        145,
        146,
        147,
    ]
    reroi = np.delete(np.arange(nregions), remove_roi)
    mregions = len(reroi)

    regions_ACA = [16, 90]
    regions_RSC = [23, 24, 97, 98]
    regions_Th = [51, 125]

    def __init__(self, data_path=".") -> None:
        self.data_path = data_path
        # self.BOLD = self.get_Bold(join(data_path, "AllenMouseConnectome", "BOLD_data.pkl"))

    def get_Bold(self, path):
        with open(path, "rb") as f:
            BOLD = pickle.load(f)
        return BOLD

    def get_SC(self):
        data_path = self.data_path
        A148 = np.loadtxt(join(data_path, "AllenMouseConnectome/Allen_148/weights.txt"))
        np.fill_diagonal(A148, 0.0)
        A148 = A148 / np.max(A148)
        return A148

    def get_roi_names(self):
        data_path = self.data_path
        with open("AllenMouseConnectome/Allen_148/region_labels.txt") as f:
            content = f.readlines()
        ROIs = [ix.strip() for ix in content]
        return ROIs

    def get_region_indices(self, flag="after_remove"):
        reroi = self.reroi
        regions_ACA = self.regions_ACA
        regions_Th = self.regions_Th
        regions_RSC = self.regions_RSC

        if flag == "after_remove":
            regions_ACA = np.where(np.isin(reroi, regions_ACA))[0]
            regions_Th = np.where(np.isin(reroi, regions_Th))[0]
            regions_RSC = np.where(np.isin(reroi, regions_RSC))[0]

        return {"ACA": regions_ACA, "Th": regions_Th, "RSC": regions_RSC}

    def get_subnetwork_indices(self, kind, division=6):

        def check_overlap(A):
            idx_labels = list(A.keys())
            idx_regions = list(A.values())
            for i in range(len(idx_regions)):
                for j in range(i + 1, len(idx_regions)):
                    if len(set(idx_regions[i]) & set(idx_regions[j])) > 0:
                        print("overlap between", idx_labels[i], idx_labels[j])

        reroi = self.reroi
        if division == 6:
            region_indices = OrderedDict([
                ('DMN', [15, 16, 17, 18, 19, 22, 23, 24, 7, 4, 69, 70, 71, 72, 73, 76, 77, 78, 61, 58]),
                ('Vis-Aud', [11, 14, 10, 12, 13, 9, 65, 68, 64, 66, 67, 63]),
                ('LCN', [1, 2, 3, 5, 6, 8, 20, 21, 25, 26, 55, 56, 57, 59, 60, 62, 74, 75, 79, 80]),
                ('BF', [27, 28, 35, 36, 37, 38, 39, 40, 41, 42, 52, 53, 0, 81, 82, 89, 90, 91, 92, 93, 94, 95, 96, 106, 107, 54]),
                ('HPF', [29, 30, 31, 32, 33, 34, 83, 84, 85, 86, 87, 88]),
                ('Th', [43, 44, 45, 46, 47, 48, 49, 50, 51, 97, 98, 99, 100, 101, 102, 103, 104, 105])
                
            ])
            
        else:
            region_indices = OrderedDict([
                ('Prefrontal', [15, 16, 17, 18, 19, 0, 69, 70, 71, 72, 73, 54]),
                ('Medial', [11, 14, 22, 23, 24, 65, 68, 76, 77, 78]),
                ('Somatosensor', [7, 4, 1, 2, 3, 5, 6, 8, 61, 58, 55, 56, 57, 59, 60, 62]),
                ('Lateral', [20, 21, 25, 26, 74, 75, 79, 80]),
                ('VisualAcoustic', [10, 12, 13, 9, 64, 66, 67, 63]),
                ('HPF', [29, 30, 31, 32, 33, 34, 83, 84, 85, 86, 87, 88]),
                ('Th', [43, 44, 45, 46, 47, 48, 49, 50, 51, 97, 98, 99, 100, 101, 102, 103, 104, 105]),
                ('CTXcp_Basal_HY', [27, 28, 35, 36, 37, 38, 39, 40, 41, 42, 52, 53, 81, 82, 89, 90, 91, 92, 93, 94, 95, 96, 106, 107])
            ])
            
        A = {}
        for key in region_indices.keys():
            A[key] = np.sort(reroi[region_indices[key]]).tolist()
        # check_overlap(A)
        if kind == "simulation":
            return A
        elif kind == "analysis":
            return region_indices
        else:
            raise ValueError("kind must be either simulation or analysis")

    def get_bold(self, group, subject):
        BOLD = self.BOLD
        return BOLD[group][subject].T


def read_pickle(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def write_bash_script(part,
                      path,
                      time="24:00:00",
                      account="paj2415",
                      script="one_batch.py",
                      env="/p/project/paj2415/VE/myvbi/bin/activate"
                      ):
    
    os.makedirs("logs", exist_ok=True)
    filename = f"logs/run_part_{part}.sh"
    with open(filename, "w") as f:
        f.write(f"#!/bin/bash\n\n")
        f.write(f"#SBATCH --account={account}\n")
        f.write(f"#SBATCH --job-name=part_{part}\n")
        f.write(f"#SBATCH --output=logs/part_{part}.out\n")
        f.write(f"#SBATCH --error=logs/part_{part}.err\n")
        f.write(f"#SBATCH --time={time}\n")
        f.write(f"#SBATCH --partition=gpus\n")
        f.write(f"#SBATCH --gres=gpu:1\n\n")
        f.write("ml Python CUDA GCC \n") # PyTorch/2.1.2 CuPy
        f.write("export MKL_NUM_THREADS=1\n")
        f.write("export NUMEXPR_NUM_THREADS=1\n")
        f.write("export OMP_NUM_THREADS=1\n")
        f.write(f"source {env}\n")
        f.write(f"python -W ignore {script} {part} {path}\n")
        
    return filename


def get_fcd_edge(bold):
    """!
    Compute the FCD from the BOLD time series.

    \param bold : array_like
        BOLD time series of shape (nnodes, ntime)

    \return array_like
            matrix of edge functional connectivity dynamics of shape (time, time)
    """

    cf = compute_cofluctuation(bold.T)  # (time, node_pairs)
    return np.var(np.corrcoef(cf))              # (time, time)


def compute_cofluctuation(ts):
    """!
    Compute co-fluctuation (functional connectivity edge) time series for
    each pair of nodes by element-wise multiplication of z-scored node time
    series.


    @param ts : array_like
        Time series of shape (time, nodes).ts

    \return array_like
        Co-fluctuation (edge time series) of shape (time, node_pairs).
    """
    nt, nn = ts.shape
    ts = stats.zscore(ts, axis=0)

    pairs = np.triu_indices(nn, 1)
    cf = ts[:, pairs[0]] * ts[:, pairs[1]]

    return cf


def get_fcd(bold, **kwargs):
    """!
    Functional Connectivity Dynamics from a collection of time series

    Parameters
    ----------
    data: np.ndarray (2d)
        time series in rows [n_nodes, n_samples]
    kwargs: dict
        parameters including:
        olap: float
            overlap between windows
        wwidth: int
            window width
        maxNwindows: int
            maximum number of windows

    Returns
    -------
    FCD: np.ndarray (2d)
        functional connectivity dynamics matrix

    """

    olap = kwargs.get("olab", 0.94)
    wwidth = kwargs.get("wwidth", 50)
    maxNwindows = kwargs.get("maxNwindows", 200)
    masks = kwargs.get("masks", {})
    assert olap <= 1 and olap >= 0, "olap must be between 0 and 1"
    nn, nt = bold.shape
    
    if masks:
        for key in masks.keys():
            mask = masks[key]
            assert mask.shape == (nn, nn), "mask shape must be equal to the number of nodes"
    
    fc_stream = []
    Nwindows = min(((nt - wwidth * olap) // (wwidth * (1 - olap)), maxNwindows))
    shift = int((nt - wwidth) // (Nwindows - 1))
    if Nwindows == maxNwindows:
        wwidth = int(shift // (1 - olap))

    indx_start = range(0, (nt - wwidth + 1), shift)
    indx_stop = range(wwidth, (1 + nt), shift)

    for j1, j2 in zip(indx_start, indx_stop):
        aux_s = bold[:, j1:j2]
        corr_mat = np.corrcoef(aux_s)
        fc_stream.append(corr_mat)
        
    fc_stream = np.asarray(fc_stream)
    mask_full = np.ones((nn, nn))
    if not masks:
        masks = {"full": mask_full}

    FCDs = {}
    for key in masks.keys():
        mask = masks[key].astype(np.float64)
        mask *= np.triu(mask_full, k=1)
        nonzero_idx = np.nonzero(mask)
        fc_stream_masked = fc_stream[:, nonzero_idx[0], nonzero_idx[1]]
        fcd = np.corrcoef(fc_stream_masked, rowvar=True)
        FCDs[key] = fcd
    return FCDs



def get_fc(ts, masks=None, positive=False, fc_fucntion="corrcoef"):

    from numpy import corrcoef

    n_noes = ts.shape[0]
    if masks is None:
        masks = {"full": np.ones((n_noes, n_noes))}

    FCs = {}
    FC = eval(fc_fucntion)(ts)
    for _, key in enumerate(masks.keys()):
        mask = masks[key]
        fc = deepcopy(FC)
        if positive:
            fc = fc * (fc > 0)
        fc = fc * mask
        fc = fc - np.diag(np.diagonal(fc))
        FCs[key] = fc

    return FCs


def set_k_diogonal(A, k, value=0.0):
    """
    set k diagonals of the given matrix to given value.
    """

    assert len(A.shape) == 2
    n = A.shape[0]
    assert k < n
    for i in range(-k, k + 1):
        a1 = np.diag(np.random.randint(1, 2, n - abs(i)), i)
        idx = np.where(a1)
        A[idx] = value
    return A


def get_fcd_edge(bold):
    """!
    Compute the FCD from the BOLD time series.

    \param bold : array_like
        BOLD time series of shape (nnodes, ntime)

    \return array_like
            matrix of edge functional connectivity dynamics of shape (time, time)
    """

    cf = compute_cofluctuation(bold.T)  # (time, node_pairs)
    return np.var(np.corrcoef(cf))  # (time, time)


def compute_cofluctuation(ts):
    """!
    Compute co-fluctuation (functional connectivity edge) time series for
    each pair of nodes by element-wise multiplication of z-scored node time
    series.


    @param ts : array_like
        Time series of shape (time, nodes).ts

    \return array_like
        Co-fluctuation (edge time series) of shape (time, node_pairs).
    """
    nt, nn = ts.shape
    ts = stats.zscore(ts, axis=0)

    pairs = np.triu_indices(nn, 1)
    cf = ts[:, pairs[0]] * ts[:, pairs[1]]

    return cf


def make_mask(n, indices):
    """
    make a mask matrix with given indices

    Parameters
    ----------
    n : int
        size of the mask matrix
    indices : list
        indices of the mask matrix

    Returns
    -------
    mask : numpy.ndarray
        mask matrix
    """
    # check validity of indices
    if not isinstance(indices, (list, tuple, np.ndarray)):
        raise ValueError("indices must be a list, tuple, or numpy array.")
    if not all(isinstance(i, (int, np.int64, np.int32, np.int16)) for i in indices):
        raise ValueError("indices must be a list of integers.")
    if not all(i < n for i in indices):
        raise ValueError("indices must be smaller than n.")

    mask = np.zeros((n, n), dtype=np.int64)
    mask[np.ix_(indices, indices)] = 1
    mask = mask - np.diag(np.diag(mask))

    return mask

