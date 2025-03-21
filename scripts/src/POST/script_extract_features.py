from helpers import *
from scipy.stats import zscore
from vbi import get_features_by_domain, get_features_by_given_names, report_cfg, update_cfg
from vbi import extract_features, extract_features_df


path = "output"
# path="/p/project/paj2415/output/1"
n_workers = 10
os.makedirs(path+"/figs", exist_ok=True)

def load_bolds(part, offset=50, nt=900, path=path):
    data = np.load(path + f"/output_{part:03d}.npz")
    bolds = data['fmri_d'].transpose(2,1,0)
    bolds = bolds[:, :, offset:-1]
    return bolds[:, :, :nt]


def preprocess(bold, nt=900):
    
    bold = bold - np.mean(bold, axis=1, keepdims=True)
    return bold[:, :nt]


def get_features(bolds, cfg, TR=0.3, verbose=True, **kwargs):
    data = extract_features_df(
        bolds, fs=1.0 / TR, cfg=cfg, verbose=verbose, **kwargs  
    )
    return data

cfg = get_features_by_domain(domain="connectivity")
cfg = get_features_by_given_names(cfg, names=[ "fc_stat", "fcd_stat"]) # "fcd_stat",
cfg = update_cfg(cfg, name="fc_stat", parameters={"features": ["sum", "mean", "std"], "eigenvalues": False})
cfg = update_cfg(cfg, name="fcd_stat", parameters={"features": ["mean", "std"], "eigenvalues": False, "win_len": 50 // 0.3})
cfg
report_cfg(cfg)


x = []
n_parts = 32
for i in range(n_parts):
	bolds = load_bolds(i)
	x_vec = get_features(bolds[:, :, :], cfg, TR=0.3, verbose=True, preprocess=preprocess, preprocess_args={}, n_workers=n_workers)
	x.append(x_vec) 

x = pd.concat(x)
x.to_csv(path+"/features.csv", index=False)

