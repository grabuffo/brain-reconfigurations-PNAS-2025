import numpy as np
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt 
from matplotlib.colors import to_hex
import matplotlib.cm as cm
import tqdm
from scipy.signal import welch
from scipy.stats import gaussian_kde
from matplotlib.cm import get_cmap
from scipy.stats import ttest_ind


import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(os.path.join(parent_dir, 'scripts'))

from Allen_Connectome_Network import *

plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams.update({
    'font.size': 10,          # General font size
    'axes.titlesize': 10,     # Title font size
    'axes.labelsize': 10,     # X and Y label font size
    'xtick.labelsize': 10,    # X tick label font size
    'ytick.labelsize': 10,    # Y tick label font size
    'legend.fontsize': 10,    # Legend font size
    'figure.titlesize': 10    # Figure title font size
})

def get_welch(ts, fs, nperseg=256):
    # ts: nt, nn
    f, Pxx = welch(ts, fs=fs, axis=0, nperseg=nperseg)
    return f, Pxx    
    
def get_eta(region_name, low_high_on, region_map):
    eta_base = region_map[region_name]["eta_base"] * coefficient_map[low_high_on]
    regions_target = region_map[region_name]["indices"]
    return eta_base, regions_target

def plot_psd(f, Pxx, ax, label, color):
    ax.plot(f, Pxx, label=label, color=color, alpha=0.3, lw=1)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power spectral density')
    ax.margins(x=0)


def clean_bold(array):
    
    # Create a mask identifying which (A, B) matrices along C contain NaNs
    nan_mask = np.isnan(array)
    
    # Remove slices along C where any NaN appears in the corresponding (A, B) matrix
    keep_mask = ~nan_mask.any(axis=(0, 1))  # Keep C indices without NaNs in (A, B)
    
    # Filter the array along the C-axis
    filtered_array = array[:, :, keep_mask]
    
    return filtered_array

def clean_psd(array):
    
    # Create a mask identifying which (B,C) matrices along A contain NaNs
    nan_mask = np.isnan(array)
    
    # Remove slices along A where any NaN appears in the corresponding (B, C) matrix
    keep_mask = ~nan_mask.any(axis=(1, 2))  # Keep A indices without NaNs in (B, C)
    
    # Filter the array along the C-axis
    filtered_array = array[keep_mask, :, :]
    
    return filtered_array

def clean_R(array):

    # Identify columns (C-axis) with NaN values
    nan_mask = np.isnan(array)
    
    # Create a mask for C slices that do NOT contain any NaNs
    keep_mask = ~nan_mask.any(axis=0)
    
    # Filter the array to keep only the valid C slices
    filtered_array = array[:, keep_mask]
    
    return filtered_array

def compute_corr_matrices(data):
    '''  
    data: 3D array of shape (time, regions, trials)
    '''

    num_trials = data.shape[2]
    corr_matrices = np.zeros((num_trials, data.shape[1], data.shape[1]))
    for i in range(num_trials):
        trial_data = data[:, :, i]  
        corr_matrix = np.corrcoef(trial_data, rowvar=False)  # (regions, regions)
        corr_matrices[i, :, :] = corr_matrix
    return corr_matrices

######################################
# COLORS
c_olh=['#808080', '#3498DB', '#FF5733']


#CMAP DIVERGENT
import matplotlib.colors as mpc
from matplotlib.colors import LinearSegmentedColormap

new_colors=[
    (0.1, 0.4, 0.6),   # Adjusted blue with deeper tone
    (0.2, 0.5, 0.7),   # Adjusted blue with balanced tone
    (0.3, 0.6, 0.8),
    (1.,1.,1.),
    (1.0, 0.8, 0.8),
    (1.0, 0.6, 0.4),   # Balanced red-orange
    (0.9, 0.4, 0.4),   # Muted deeper red
]


extended_portions= [0.,0.2,0.35,0.5,0.65,0.8, 1.]

new_cmap_name='delta_divergent'

new_cmap=LinearSegmentedColormap.from_list(
    new_cmap_name, list(zip(extended_portions,new_colors))
)

##############################################################
# low is inhibition
# high is excitation
# on is baseline

coefficient_map = {
    "low": 1.2,
    "on": 1.0,
    "high": 0.8,
}

region_map = {
    "RSC": {"indices": [23, 24, 97, 98], "eta_base": -4.184},
    "ACAv": {"indices": [16, 90], "eta_base": -4.184},
    "MTH": {"indices": [45, 51, 119, 125], "eta_base": -3.725},
    "PFC": {"indices": [15, 16, 17, 89, 90, 91], "eta_base": -4.184},
    "SS": {"indices": [4, 7, 78, 81], "eta_base": -4.184},
    "VIS": {"indices": [11, 14, 85, 88], "eta_base": -4.957},
    "HYP": {"indices": [53, 54, 127, 128], "eta_base": -3.757},
    "HPC": {"indices": [30, 31, 32], "eta_base": -4.011},
    "TP": {"indices": [25, 26], "eta_base": -3.581},
}