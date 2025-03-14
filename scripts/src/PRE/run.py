from helpers import *

seed = 2
np.random.seed(seed)
torch.manual_seed(seed)

ns = 100_000
chunk_size = 10_000
dtype = "float"
# engine = "cpu"
engine = "gpu"
path = "/p/project/paj2415/output/"
os.makedirs(path, exist_ok=True)

DS = G_Dataset(data_path="../../../data/")
weights = DS.get_SC()
nn = weights.shape[0]
reroi = DS.reroi.tolist()

idx_regions_dict = DS.get_subnetwork_indices(kind="simulation", division=6)
idx_labels = list(idx_regions_dict.keys())
idx_regions = list(idx_regions_dict.values())
theta_eta = -4.6 * np.ones(nn).astype(dtype)
theta_G = np.random.uniform(0.0, 0.5, ns)
theta_eta = np.random.uniform(-6.0, -3.5, size=(6, ns))
eta = np.ones((nn, ns)) * -4.6

for i in range(ns):
    for j in range(6):
        eta[idx_regions[j], i] = theta_eta[j, i]

par = {
    "G": theta_G,        # global coupling strength
    "weights": weights,  # connection matrix
    "method": "heun",    # integration method
    "t_cut": 0,
    "dt": 0.01,
    "t_end": 300_000,    # [ms]
    "num_sim": chunk_size,
    "tr": 300.0,
    "eta": eta,
    "engine": engine,    # cpu or gpu
    "seed": seed,        # seed for random number generator
    "RECORD_RV": False,
    "RECORD_BOLD": True,
    "same_initial_state": True,
    "output": path,
}


# store par in pickle file
with open(path + "/par.pkl", "wb") as f:
    pickle.dump(par, f)

G_chunks = np.array_split(par["G"], ns // chunk_size)
eta_chunks = np.array_split(par["eta"], ns // chunk_size, axis=1)
eta6_chunks = np.array_split(theta_eta, ns // chunk_size, axis=1)

with open(path + "/priors.npz", "wb") as f:
    np.savez(f, G=G_chunks, eta=eta_chunks, eta6=eta6_chunks)

num_chunks = ns // chunk_size

for i in range(num_chunks):
    # run local:
    # os.system(f"python one_batch.py {i} {path}")

    # run on cluster:
    job_file = write_bash_script(i, path)
    os.system(f"sbatch {job_file}")
