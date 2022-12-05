import numpy as np
import matplotlib.pyplot as plt


def get_sem(data):
    """
    returns standard error of the mean
    """
    return np.std(data, axis=1, ddof=1) / np.sqrt(np.size(data, axis=1))


print("phase diagram")
dir_name = "221206-sim-7"

probs = np.loadtxt(dir_name + "/probs.csv")
ps = probs[:, 1]
print(ps)

sim_num = np.loadtxt(dir_name + "/sim_num.csv", dtype=int)
print(sim_num)

t_start = 800
cs = np.zeros((len(ps), sim_num))
xs = np.zeros((len(ps), sim_num))
print(cs.shape)
plt.figure(1)
for i in range(len(ps)):
    for sim_number in range(sim_num):
        single_sim = np.loadtxt(dir_name + f"/{i}/sim-{sim_number}.txt")
        single_sim = single_sim[t_start:]
        single_sim2 = [(2 * x - 1) ** 2 for x in single_sim]
        single_sim = [max(x, 1 - x) for x in single_sim]
        cs[i, sim_number] = np.mean(single_sim)
        xs[i, sim_number] = np.mean(single_sim2) - (2 * cs[i, sim_number] - 1) ** 2
        plt.plot(single_sim)
        print(np.mean(single_sim))

print(get_sem(cs))
plt.figure(2)
plt.plot(ps, cs, ".-")
plt.xlabel("nonconformity probability")
plt.ylabel("concentration")

plt.figure(3)
plt.errorbar(ps, np.mean(cs, axis=1), yerr=get_sem(cs), fmt='.-')
plt.xlabel("nonconformity probability")
plt.ylabel("concentration")

plt.figure(4)
plt.errorbar(ps, np.mean(xs, axis=1), yerr=get_sem(xs), fmt='.-')
plt.xlabel("nonconformity probability")
plt.ylabel("fluctuation")
plt.show()
