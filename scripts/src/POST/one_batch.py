from helpers import *


if len(sys.argv) > 2:
    pass
else:
    print("Usage: python single.py <part:int> <path:string>")
    sys.exit()

part = int(sys.argv[1])
path = sys.argv[2]

DS = G_Dataset()
weights = DS.get_SC()
nn = weights.shape[0]
reroi = DS.reroi.tolist()

# priors = np.load(path + "/priors.npz")
priors = read_pickle(path + "/priors.pkl")
Gs = priors["G"][part]
eta = priors["eta"][part]
eta1 = priors["eta1"][part]

par = read_pickle(path + "/par.pkl")
par["G"] = Gs
par["eta"] = eta
par["num_sim"] = len(Gs)
assert eta.shape[0] == nn, "eta shape mismatch"

sde = MPR_sde(par)
sol = sde.run()

fmri_d = sol["fmri_d"][:, reroi, :]
fmri_t = sol["fmri_t"]

# print(fmri_d.shape, fmri_t.shape)

np.savez(
    path + "/output_{:03d}.npz".format(part),
    fmri_d=fmri_d,
    fmri_t=fmri_t,
    G=Gs,
    eta=eta,
    eta1=eta1
)
