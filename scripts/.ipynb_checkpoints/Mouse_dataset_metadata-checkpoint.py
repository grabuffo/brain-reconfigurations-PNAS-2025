import os
import sys

from F1_functions import *
from Allen_Connectome_metadata import *

# tlen_kai=2000 # number of time points in DREADDs dataset
# tlen_arm=512 # number of time points in Lesion dataset


# Experimental Subjects
SUBS={'ACA':['Wt8648', 'Wt8656', 'Wt8657', 'Wt8650', 'Wt8652', 'Wt8653', 'Wt8658'],
      'RSC':['Wt8943', 'Wt8944', 'Wt8942', 'Wt8934', 'Wt8938', 'Wt8940', 'Wt8936', 'Wt8941', 'Wt8639', 'Wt8638', 'Wt8642', 'Wt8640', 'Wt8647', 'Wt8637'],
      'CTRL':['Wt8919', 'Wt8928', 'Wt8918', 'Wt8923', 'Wt8925', 'Wt8921', 'Wt8926', 'Wt8917'],
      'Th':['F1R1V', 'F2R2V', 'F3R', 'F3R3V', 'M2B', 'M3B3R']} # We exclude mice because they do not have lesions visible 'F1R', 'M3B'


# Association of subjects and groups (CTRL, RSC, ACA, Th)

KEYS={'ACA': ['Wt8648_Baseline',
  'Wt8650_Baseline',
  'Wt8652_Baseline',
  'Wt8653_Baseline',
  'Wt8656_Baseline',
  'Wt8657_Baseline',
  'Wt8658_Baseline',
  'Wt8648_CNO',
  'Wt8650_CNO',
  'Wt8652_CNO',
  'Wt8653_CNO',
  'Wt8656_CNO',
  'Wt8657_CNO',
  'Wt8658_CNO'],
 'RSC': ['Wt8637_Baseline',
  'Wt8638_Baseline',
  'Wt8639_Baseline',
  'Wt8640_Baseline',
  'Wt8642_Baseline',
  'Wt8647_Baseline',
  'Wt8934_Baseline',
  'Wt8936_Baseline',
  'Wt8938_Baseline',
  'Wt8940_Baseline',
  'Wt8941_Baseline',
  'Wt8942_Baseline',
  'Wt8943_Baseline',
  'Wt8944_Baseline',
  'Wt8637_CNO',
  'Wt8638_CNO',
  'Wt8639_CNO',
  'Wt8640_CNO',
  'Wt8642_CNO',
  'Wt8647_CNO',
  'Wt8934_CNO',
  'Wt8936_CNO',
  'Wt8938_CNO',
  'Wt8940_CNO',
  'Wt8941_CNO',
  'Wt8942_CNO',
  'Wt8943_CNO',
  'Wt8944_CNO'],
 'CTRL': ['Wt8917_Baseline',
  'Wt8918_Baseline',
  'Wt8919_Baseline',
  'Wt8921_Baseline',
  'Wt8923_Baseline',
  'Wt8925_Baseline',
  'Wt8926_Baseline',
  'Wt8928_Baseline',
  'Wt8917_CNO',
  'Wt8918_CNO',
  'Wt8919_CNO',
  'Wt8921_CNO',
  'Wt8923_CNO',
  'Wt8925_CNO',
  'Wt8926_CNO',
  'Wt8928_CNO'],
 'Th': ['F1R1V',
  'F2R2V',
  'F3R',
  'F3R3V',
  'M2B',
  'M3B3R',
  'F1R1V_post',
  'F2R2V_post',
  'F3R_post',
  'F3R3V_post',
  'M2B_post',
  'M3B3R_post']}


# Import Global Connectivity values (GC) and Global Metastability (GM)

GC={'ACA':{}, 'RSC':{}, 'CTRL':{}, 'Th':{}}
for filename in os.listdir(path_data_DREADDs):
    if filename.startswith("GC_DMN_A24_CNO"):
        GC['ACA'][filename.strip('GC_DMN_A24_CNO_').strip('.npy')]=np.load(path_data_DREADDs+filename)
for filename in os.listdir(path_data_DREADDs):
    if filename.startswith("GC_DMN_A30_CNO"):
        GC['RSC'][filename.strip('GC_DMN_A30_CNO_').strip('.npy')]=np.load(path_data_DREADDs+filename)
for filename in os.listdir(path_data_DREADDs):
    if filename.startswith("GC_DMN_A29c_CNO"):
        GC['RSC'][filename.strip('GC_DMN_A29c_CNO_').strip('.npy')]=np.load(path_data_DREADDs+filename)
for filename in os.listdir(path_data_DREADDs):
    if filename.startswith("GC_DMN_A30_control_CNO"):
        GC['CTRL'][filename.strip('GC_DMN_A30_control_CNO_').strip('.npy')]=np.load(path_data_DREADDs+filename)
        
for filename in os.listdir(path_data_lesion):
    if filename.startswith("GC"):
        GC['Th'][my_removeprefix(filename,'GC_').strip('.npy')]=np.load(path_data_lesion+filename)
        
GM={'ACA':{}, 'RSC':{}, 'CTRL':{}, 'Th':{}}
for filename in os.listdir(path_data_DREADDs):
    if filename.startswith("GM_DMN_A24_CNO"):
        GM['ACA'][filename.strip('GM_DMN_A24_CNO_').strip('.npy')]=np.load(path_data_DREADDs+filename)
for filename in os.listdir(path_data_DREADDs):
    if filename.startswith("GM_DMN_A30_CNO"):
        GM['RSC'][filename.strip('GM_DMN_A30_CNO_').strip('.npy')]=np.load(path_data_DREADDs+filename)
for filename in os.listdir(path_data_DREADDs):
    if filename.startswith("GM_DMN_A29c_CNO"):
        GM['RSC'][filename.strip('GM_DMN_A29c_CNO_').strip('.npy')]=np.load(path_data_DREADDs+filename)
for filename in os.listdir(path_data_DREADDs):
    if filename.startswith("GM_DMN_A30_control_CNO"):
        GM['CTRL'][filename.strip('GM_DMN_A30_control_CNO_').strip('.npy')]=np.load(path_data_DREADDs+filename)
        
for filename in os.listdir(path_data_lesion):
    if filename.startswith("GM"):
        GM['Th'][my_removeprefix(filename,'GM_').strip('.npy')]=np.load(path_data_lesion+filename)

GC_vox={'ACA':np.zeros((len(SUBS['ACA']),2,np.sum(msk))), 
    'RSC':np.zeros((len(SUBS['RSC']),2,np.sum(msk))),
    'CTRL':np.zeros((len(SUBS['CTRL']),2,np.sum(msk))),'Th':np.zeros((len(SUBS['Th']),2,np.sum(MASK)))}
GM_vox={'ACA':np.zeros((len(SUBS['ACA']),2,np.sum(msk))), 
    'RSC':np.zeros((len(SUBS['RSC']),2,np.sum(msk))),
    'CTRL':np.zeros((len(SUBS['CTRL']),2,np.sum(msk))),'Th':np.zeros((len(SUBS['Th']),2,np.sum(MASK)))}
for i in GC_vox.keys():
    for ij, j in enumerate(KEYS[i]):
        Gc=GC[i][j]
        Gm=GM[i][j]
        if ij<len(SUBS[i]):
            GC_vox[i][ij,0,:]=Gc
            GM_vox[i][ij,0,:]=Gm
        else:
            GC_vox[i][ij-len(SUBS[i]),1,:]=Gc
            GM_vox[i][ij-len(SUBS[i]),1,:]=Gm


# Import Global Connectivity values (GC) and Global Metastability (GM) with ROIs and RSNs details

d_gc_pre={}
d_gc_post={}
for i in ['ACA', 'RSC', 'CTRL', 'Th']:
    d_gc_pre[i]={}
    d_gc_post[i]={}
    for j in ['DMN', 'Vis', 'LCN', 'BF', 'HPF', 'Th']:
        d_gc_pre[i][j]={}
        d_gc_post[i][j]={}
        for k in RSNs[j]:
            VOX=VOX_rois[idt(k)]
            vox=vox_rois[idt(k)]
            if i=='Th':
                d_gc_pre[i][j][ROIsm[k]]=np.mean(GC_vox[i][:,0,np.where(VOX==1)],axis=0)
                d_gc_post[i][j][ROIsm[k]]=np.mean(GC_vox[i][:,1,np.where(VOX==1)],axis=0)  
            else:
                d_gc_pre[i][j][ROIsm[k]]=np.mean(GC_vox[i][:,0,np.where(vox==1)],axis=0)
                d_gc_post[i][j][ROIsm[k]]=np.mean(GC_vox[i][:,1,np.where(vox==1)],axis=0)

d_gm_pre={}
d_gm_post={}
for i in ['ACA', 'RSC', 'CTRL', 'Th']:
    d_gm_pre[i]={}
    d_gm_post[i]={}
    for j in ['DMN', 'Vis', 'LCN', 'BF', 'HPF', 'Th']:
        d_gm_pre[i][j]={}
        d_gm_post[i][j]={}
        for k in RSNs[j]:
            VOX=VOX_rois[idt(k)]
            vox=vox_rois[idt(k)]
            if i=='Th':
                d_gm_pre[i][j][ROIsm[k]]=np.mean(GM_vox[i][:,0,np.where(VOX==1)],axis=0)
                d_gm_post[i][j][ROIsm[k]]=np.mean(GM_vox[i][:,1,np.where(VOX==1)],axis=0)  
            else:
                d_gm_pre[i][j][ROIsm[k]]=np.mean(GM_vox[i][:,0,np.where(vox==1)],axis=0)
                d_gm_post[i][j][ROIsm[k]]=np.mean(GM_vox[i][:,1,np.where(vox==1)],axis=0)

# Average GC and GM
GM_ACA_pre=np.mean(GM_vox['ACA'],axis=0)[0]
GM_ACA_post=np.mean(GM_vox['ACA'],axis=0)[1]
GM_RSC_pre=np.mean(GM_vox['RSC'],axis=0)[0]
GM_RSC_post=np.mean(GM_vox['RSC'],axis=0)[1]
GM_CTRL_pre=np.mean(GM_vox['CTRL'],axis=0)[0]
GM_CTRL_post=np.mean(GM_vox['CTRL'],axis=0)[1]

GC_ACA_pre=np.mean(GC_vox['ACA'],axis=0)[0]
GC_ACA_post=np.mean(GC_vox['ACA'],axis=0)[1]
GC_RSC_pre=np.mean(GC_vox['RSC'],axis=0)[0]
GC_RSC_post=np.mean(GC_vox['RSC'],axis=0)[1]
GC_CTRL_pre=np.mean(GC_vox['CTRL'],axis=0)[0]
GC_CTRL_post=np.mean(GC_vox['CTRL'],axis=0)[1]

GC_Th_pre=np.mean(GC_vox['Th'],axis=0)[0]
GC_Th_post=np.mean(GC_vox['Th'],axis=0)[1]
GM_Th_pre=np.mean(GM_vox['Th'],axis=0)[0]
GM_Th_post=np.mean(GM_vox['Th'],axis=0)[1]




