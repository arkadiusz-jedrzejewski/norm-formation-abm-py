import numpy as np
import matplotlib.pyplot as plt
from theoretical_module import get_fixed_points, get_fixed_points_q_voter, get_fixed_points_uniform


def get_sem(data):
    """
    returns standard error of the mean
    """
    return np.std(data, axis=1, ddof=1) / np.sqrt(np.size(data, axis=1))


dir_name = "221215-sim-2"

probs = np.loadtxt(dir_name + "/probs.csv", delimiter=",")
ps = probs[:, 1]

sim_num = np.loadtxt(dir_name + "/sim_num.csv", dtype=int)
params = np.loadtxt(dir_name + "/params.csv", delimiter=",")

p_start, p_end, p_num = params[0], params[1], int(params[2])
q, f, = params[4], params[5]
time_horizon, system_size = int(params[6]), int(params[7])
is_annealed = int(params[8])

time_start = 800

form = '<15'
print(f"{'loaded data:':{form}}{dir_name}",
      f"{'p_start:':{form}}{p_start}",
      f"{'p_end:':{form}}{p_end}",
      f"{'p_num:':{form}}{p_num}",
      f"{'system_size:':{form}}{system_size}",
      f"{'is_annealed:':{form}}{is_annealed}",
      f"{'q:':{form}}{q}",
      f"{'f:':{form}}{f}",
      f"{'time_horizon:':{form}}{time_horizon}",
      f"{'time_start:':{form}}{time_start}",
      f"{'time_average:':{form}}{time_horizon - time_start + 1}",
      f"{'sim_num:':{form}}{sim_num}",
      sep="\n")

cs = np.zeros((len(ps), sim_num))
xs = np.zeros((len(ps), sim_num))
us = np.zeros((len(ps), sim_num))

# plt.figure(1)
for i in range(len(ps)):
    for sim_number in range(sim_num):
        single_sim = np.loadtxt(dir_name + f"/{i}/sim-{sim_number}.txt")
        single_sim = single_sim[time_start:]
        single_sim2 = [(2 * x - 1) ** 2 for x in single_sim]
        single_sim4 = [(2 * x - 1) ** 4 for x in single_sim]
        single_sim = [max(x, 1 - x) for x in single_sim]
        cs[i, sim_number] = np.mean(single_sim)
        xs[i, sim_number] = np.mean(single_sim2) - (2 * cs[i, sim_number] - 1) ** 2
        us[i, sim_number] = 1 - np.mean(single_sim4) / (3 * np.mean(single_sim2) ** 2)
        # plt.plot(single_sim)

# plt.figure(2)
# plt.plot(ps, cs, ".-")
# plt.xlabel("nonconformity probability")
# plt.ylabel("concentration")
#

ps_theo, cs_theo = get_fixed_points(num=100, q=q, f=f, is_quenched=not is_annealed)

plt.figure(3)
plt.plot(ps_theo, cs_theo, 'r')
plt.errorbar(ps, np.mean(cs, axis=1), yerr=get_sem(cs), fmt='.')
plt.xlim([0, 1])
plt.xlabel("nonconformity probability")
plt.ylabel("concentration")
#
# plt.figure(4)
# plt.errorbar(ps, np.mean(xs, axis=1), yerr=get_sem(xs), fmt='.-')
# plt.xlabel("nonconformity probability")
# plt.ylabel("fluctuation")
#
# plt.figure(5)
# plt.errorbar(ps, np.mean(us, axis=1), yerr=get_sem(us), fmt='.-')
# plt.xlabel("nonconformity probability")
# plt.ylabel("Binder cumulant")

plt.show()
