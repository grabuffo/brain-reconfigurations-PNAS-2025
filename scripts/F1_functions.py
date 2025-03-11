import os
import sys
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt 
import nibabel as nib
from numpy.ma import masked_array
from scipy import ndimage
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

##################
# PATHS
path_allen='../data/AllenMouseConnectome/'
path_fig='../figures/Figure1/'
path_data_lesion='../data/Empirical_data_features/dynamicGlobalConnectivity/Lesion/'
path_data_DREADDs='../data/Empirical_data_features/dynamicGlobalConnectivity/DREADDs/'

##################
# FUNCTIONS

def my_removeprefix(s, prefix):
    if s.startswith(prefix):
        return s[len(prefix):]
    else:
        return s

def enlarger(fmri_image):
    fmri_image = np.repeat(fmri_image, repeats=2, axis=0)
    fmri_image = np.repeat(fmri_image, repeats=2, axis=1)
    larger_fmri_image = np.repeat(fmri_image, repeats=2, axis=2)
    return larger_fmri_image

def load_obj(name):
    with open(path_allen + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def fdr(p_vals):

    from scipy.stats import rankdata
    ranked_p_values = rankdata(p_vals)
    fdr = p_vals * len(p_vals) / ranked_p_values
    fdr[fdr > 1] = 1

    return fdr

#######################
# COLORS

# fix colors for RSNs
CRSN = {
    'DMN': '#ff7f0e',
    'Vis': '#2ca02c',
    'LCN': '#1f77b4',
    'BF': '#1A202E',
    'HPF': '#9467bd',
    'Th': '#d62728'
}

#CRSN={0:'#ff7f0e',1:'#2ca02c',2:'#1f77b4',3:'#1A202E',4:'#9467bd',5:'#d62728'}

#######################
# BRAIN HEATMAPS

def plot_brain(vol_img, Template, Mask, title, cbar_label, vmin, vmax, cmap='jet', background='off', view='coronal'):
    if len(vol_img)==30965: # Number of voxels in DREADDs data
        MS_plot=np.copy(Mask)/100
        MS_plot[np.where(Mask==1)]=vol_img
        MS_PLT=enlarger(MS_plot)
    elif len(vol_img)>30965:  #for the higher resolution of the Lesion data (237474 voxels)
        MS_plot=np.copy(Mask)/100
        MS_plot[np.where(Mask==1)]=vol_img
        MS_PLT=MS_plot
    else:
        print('unkown data')

    if view=='coronal':
        va=Template[:,::-1,::-1][:,71,:].T # selected slice 71
        vb=MS_PLT[:,::-1,::-1][:,71,:].T
    elif view=='sagittal':
        va=Template[:,::-1,::-1][59,:,:].T # selected slice 59
        vb=MS_PLT[:,::-1,::-1][59,:,:].T
    elif view=='surface':
        va=surf(Template,A148,-1).T
        vb=surf(MS_PLT,MS_PLT).T
    else:
        print('unkown view')

    v1a = masked_array(va,va<=4.5,fill_value=np.NaN)
    v1b = masked_array(vb,vb==0)#((vb<0.1)&(vb>-0.05))
    
    fig,ax = plt.subplots(figsize=(3.5,2.5))
    plt.title(title,fontsize=16)
    if background=='on':
        ax.imshow(np.where(v1a>4.5,1,np.NaN),interpolation='nearest',cmap=cm.gray,vmin=1.,vmax=1)
    else:
        pass
    cax = ax.imshow(v1b,interpolation='nearest',cmap=cmap,vmin=vmin,vmax=vmax)


    divider = make_axes_locatable(ax)
    cbar_ax = divider.append_axes("left", size="3%", pad=0.05)
    cbar = fig.colorbar(cax, cax=cbar_ax,shrink=0.35)
    cbar_ax.yaxis.set_ticks_position('left')  # Ensure ticks are on the left side
    
    # Adjust the fontsize of the colorbar ticks
    cbar.ax.tick_params(labelsize=12)
    
    # Invert x-axis of the colorbar to face left
    cbar_ax.yaxis.set_label_position('left')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar.set_label(cbar_label,fontsize=16)
    cbar_ax.set_aspect(16)
    plt.tight_layout()
    plt.savefig(path_fig +title+'.pdf',dpi=400)


#######################
# Plot bars

def bars(p, bottom, top):
    # Get info about y-axis
    yrange = top - bottom

    # Columns corresponding to the datasets of interest
    x1 = 1
    x2 = 2
    # What level is this bar among the bars above the plot?
    level = 1
    # Plot the bar
    bar_height = (yrange * 0.08 * level)+ top/2
    bar_tips = bar_height - (yrange * 0.02)
    plt.plot(
        [x1, x1, x2, x2],
        [bar_tips, bar_height, bar_height, bar_tips], lw=1, c='k')
    # Significance level

    if p < 0.001:
        sig_symbol = '*'
    elif p < 0.01:
        sig_symbol = '*'
    elif p < 0.05:
        sig_symbol = '*'
    else:
        sig_symbol = ''
    text_height = bar_height + (yrange * 0.01)
    plt.text((x1 + x2) * 0.5, text_height, sig_symbol, ha='center', c='k')
    plt.ylim((-0.09,0.07))


