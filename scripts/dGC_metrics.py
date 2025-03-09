import numpy as np
from scipy import stats

###########################################################
# EXAMPLE: Given data with shape data.shape~(time, regions)

tlen=data.shape[0]
nregions=data.shape[1]

# z-score the signals for each voxel
z=stats.zscore(data,axis=0,nan_policy='omit')            


# Global Signal
GS=np.mean(z ,axis=1)  

# Network Dynamic Profile
NDP=GS**2

# dynamic Global Connectivity
dGC=np.empty_like(data)
for i in range(nregions):
    dGC[:,i]=z[:,i]*GS

# Global Connectivity (GC)
GC=np.mean(dGC, axis=0)

# Global Metastability (GM)
GM=np.var(dGC, axis=0)


# Edge coactivation time series
def go_edge(data):
    nregions=data.shape[1]
    iTriup= np.triu_indices(nregions,k=1) 
    gz=stats.zscore(data)
    Eseries = gz[:,iTriup[0]]*gz[:,iTriup[1]]
    return Eseries

edg = go_edge(data)             

# Dynamic Functional Connectivity (time-resolved)
dFC = np.corrcoef(edg) 

# Functional Connectivity
FC = np.corrcoef(data.T) 






