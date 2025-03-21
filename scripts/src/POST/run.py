from helpers import *

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

ns_per_subj = 1024 * 4
ns = 29 * ns_per_subj
chunk_size = 4096
dtype = "float"
# engine = "cpu"
engine = "gpu"
data_path = "../../../data/SBI/POST/"
os.makedirs("output", exist_ok=True)

DS = G_Dataset(data_path="../../../data/")
weights = DS.get_SC()
nn = weights.shape[0]
reroi = DS.reroi.tolist()

idx_regions_dict = DS.get_subnetwork_indices(kind="simulation", division=6)
idx_labels = list(idx_regions_dict.keys())
idx_regions = list(idx_regions_dict.values())
theta_eta = -4.6 * np.ones(nn).astype(dtype)

# load optimal G and 6-eta for each subject from PRE:
df = pd.read_csv(data_path+"peaks_ks_opt.csv")
target_roi = {
    "RSC": DS.regions_RSC, 
    "ACA": DS.regions_ACA, 
    "CTRL": DS.regions_RSC, 
    "Th": DS.regions_Th
    }
n_subjects = {"RSC": 14, "ACA": 7, "CTRL": 8}


def get_g_eta(df, group, subject_id, target_roi=[], ns=5):
    '''
    ns: number of simulations per subject
    '''
    
    g = df[(df["group"] == group) & (df["subject_id"] == subject_id)]["g"].values[0]
    eta = df[(df["group"] == group) & (df["subject_id"] == subject_id)][['eta1', 'eta2', 'eta3', 'eta4', 'eta5', 'eta6']]
    eta.columns = ['DMN', 'Vis-Aud', 'LCN', 'BF', 'HPF', 'Th']
    eta_dict = eta.to_dict(orient='records')[0]
    
    DS = G_Dataset(".")
    nn = DS.get_SC().shape[0]
    idx_regions_dict = DS.get_subnetwork_indices(kind="simulation", division=6)
    idx_regions = list(idx_regions_dict.values())
    eta_6_roi = list(eta_dict.values())
    eta0 = -4.6 * np.ones(nn).astype("float")
    for i in range(len(idx_regions)):
        eta0[idx_regions[i]] = eta_6_roi[i]
        
    # tile eta0 to ns
    # ns = theta_eta.shape[0]
    theta_eta = np.random.uniform(-6.0, -3.5, size=(ns))
    eta = np.tile(eta0, (ns, 1)).T
    for i in range(ns):
        eta[target_roi, i] = theta_eta[i]
    g = g * np.ones(ns)
        
    return g, eta, theta_eta

Gs = []
etas = []
theta_eta = []
for gr in ["RSC", "ACA", "CTRL"]:
    n = n_subjects[gr] 
    for i in range(n):
        g, eta, theta_ = get_g_eta(df, gr, i, target_roi[gr], ns=ns_per_subj)
        
        # print(eta.shape) # 148 by 10
        Gs.extend(g)
        etas.append(eta)
        theta_eta.extend(theta_)
        
        
Gs = np.array(Gs)
etas = np.concatenate(etas, axis=1)
theta_eta = np.array(theta_eta)
# print(Gs.shape, etas.shape, theta.shape)

par = {
    "G": 0.5,               # global coupling strength
    "weights": weights,     # connection matrix
    "method": "heun",       # integration method
    "t_cut": 0,   
    "dt": 0.01,
    "t_end": 300_000,       # [ms]
    "num_sim": chunk_size,
    "tr": 300.0,
    "eta": -4.6,
    "engine": engine,       # cpu or gpu
    "seed": seed,           # seed for random number generator
    "RECORD_RV": False,
    "RECORD_BOLD": True,
    "same_initial_state": True,
    "output": "output",
}


# store par in pickle file
with open("output/par.pkl", "wb") as f:
    pickle.dump(par, f)

G_chunks = np.array_split(Gs, ns//chunk_size)
eta_chunks = np.array_split(etas, ns//chunk_size, axis=1)
eta1_chunks = np.array_split(theta_eta, ns//chunk_size)
num_chunks = ns//chunk_size

with open("output/priors.pkl", "wb") as f:
    pickle.dump({"G": G_chunks, "eta": eta_chunks, "eta1": eta1_chunks}, f)


for i in range(num_chunks):
    # run local:
    print(f"Running chunk {i+1} / {num_chunks}")
    os.system(f"python one_batch.py {i} {"output"}")
    
    # run on cluster:
    # job_file = write_bash_script(i, path)
    # os.system(f"sbatch {job_file}")
    
