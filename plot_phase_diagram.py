import numpy as np

print("phase diagram")
dir_name = "221205-sim-1"

probs = np.loadtxt(dir_name + "/probs.csv")
ps = probs[:, 1]
print(ps)

sim_num = np.loadtxt(dir_name + "/sim_num.csv", dtype=int)
print(sim_num)

