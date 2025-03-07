import os
import sys

from F1_functions import *

###################################################
# IMPORT ALLEN DATA

with open(path_allen+'Allen_148/region_labels.txt') as f:
    content = f.readlines()  

ROIs = [ix.strip() for ix in content] # ROIs names

volsize=np.loadtxt(path_allen+'/Allen_148/volsize.txt') # ROIs vlumes

nregions=len(ROIs) # number of ROIs

file_refs= nib.load(path_allen + 'Template_rotated_Allen.nii') # ALLEN TEMPLATE
Template= file_refs.get_fdata() 

file_refs= nib.load(path_allen + 'Vol_148_Allen.nii')
A148= file_refs.get_fdata() # Allen volumes
a148=A148[::2,::2,::2] # coarse grained Allen volumes

# remove ROIs that present artifacts in fMRI data
remove_roi=[27, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 
            101, 129, 130,  131,  132,  133,  134,  135,  136,  137,  138,  139,  140, 141, 142, 143,144,145,146,147]
reroi=np.delete(np.arange(nregions),remove_roi)
mregions=len(reroi) # filtered number of ROIs

ROIsm=np.asarray(ROIs)[reroi] # filtered ROIs names


######################################################
# Resting State Network Organization
# RSN to ROIs ID association

RSNs={'DMN': [15, 16, 17, 18, 19, 22, 23, 24, 7, 4, 69, 70, 71, 72, 73, 76, 77, 78, 61, 58], 
 'Vis': [11, 14, 10, 12, 13, 9, 65, 68, 64, 66, 67, 63], 
 'LCN': [1, 2, 3, 5, 6, 8, 20, 21, 25, 26, 55, 56, 57, 59, 60, 62, 74, 75, 79, 80], 
 'BF': [27, 28, 35, 36, 37, 38, 39, 40, 41, 42, 52, 53, 0, 81, 82, 89, 90, 91, 92, 93, 94, 95, 96, 106, 107, 54], 
 'HPF': [29, 30, 31, 32, 33, 34, 83, 84, 85, 86, 87, 88], 
 'Th': [43, 44, 45, 46, 47, 48, 49, 50, 51, 97, 98, 99, 100, 101, 102, 103, 104, 105]}

RSNs_mod=list(RSNs.keys())

lenmods=[len(RSNs['DMN']),len(RSNs['Vis']),len(RSNs['LCN']),len(RSNs['BF']),len(RSNs['HPF']),len(RSNs['Th'])]

order_mod=np.asarray(RSNs['DMN']+RSNs['Vis']+RSNs['LCN']+RSNs['BF']+RSNs['HPF']+RSNs['Th'])

def idt(d):
    return np.where(np.asarray(ROIs) == ROIsm[d])[0][0]


#######################
# MASKS AND VOLUMES

# mask Allen (the DREADDs data is already aligned to the Allen Mask)
mask=np.copy(A148)
for i in remove_roi:
    mask=np.where(mask==i,-1,mask)
mask=np.where(mask!=-1,1,-1)
mask=np.where(mask!=-1,1,0)
msk=mask[::2,::2,::2]# coarse grained mask

# mask Lesion data
MASK_LESION=load_obj('MASK_LESION')  # Mask of single mice in Lesion Dataset
MASK=np.asarray(list(MASK_LESION.values()))
MASK=np.sum(MASK,axis=0)
MASK=np.where(MASK==12,1,0)

# ROI to volume correspondence 
VOL_rois={} # fine grained roi to volume dict
vol_rois={} # coarse grained roi to volume dict
for i in range(148):
    VOL_rois[i]=len(np.where(A148==i)[0])
    vol_rois[i]=len(np.where(a148==i)[0])

# ROI to volume correspondence 
VOX_rois={}
vox_rois={}
for i in range(148):
    MST=np.copy(MASK); mst=np.copy(msk)
    MST[np.where(A148==i)]=2*np.ones(VOL_rois[i])
    mst[np.where(a148==i)]=2*np.ones(vol_rois[i])
    IDX=MST[np.where(MASK==1)]
    idx=mst[np.where(msk==1)]
    VOX_rois[i]=np.where(IDX==2,1,0)
    vox_rois[i]=np.where(idx==2,1,0)

# RSN to Voxel association
VOX_rsn={}
vox_rsn={}
for i in list(RSNs.keys()):
    VOX_rsn[i]=np.zeros(237474)    
    vox_rsn[i]=np.zeros(30965)
    for j in RSNs[i]:
        VOX_rsn[i]+=VOX_rois[idt(j)]
        vox_rsn[i]+=vox_rois[idt(j)]