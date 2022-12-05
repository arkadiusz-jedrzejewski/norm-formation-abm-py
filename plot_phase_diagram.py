import numpy as np
import matplotlib.pyplot as plt

print("phase diagram")
dir_name = "221205-sim-1"

probs = np.loadtxt(dir_name + "/probs.csv")
ps = probs[:, 1]
print(ps)

sim_num = np.loadtxt(dir_name + "/sim_num.csv", dtype=int)
print(sim_num)

cs = np.zeros(ps.shape)
plt.figure(1)
for i in range(len(ps)):
    single_sim = np.loadtxt(dir_name + f"/{i}/sim-{1}.txt")
    cs[i] = max(single_sim[-1], 1 - single_sim[-1])
    plt.plot(single_sim)
    print(single_sim)

print(cs)
plt.figure(2)
plt.plot(ps, cs, ".-")
plt.xlabel("nonconformity probability")
plt.ylabel("concentration")
plt.show()

