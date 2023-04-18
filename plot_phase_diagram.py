import numpy as np
import matplotlib.pyplot as plt
from theoretical_module import plot_fixed_points_p, plot_fixed_points_k, plot_fixed_points_m, plot_fixed_points_x0


def get_sem(data):
    """
    returns standard error of the mean
    """
    return np.std(data, axis=1, ddof=1) / np.sqrt(np.size(data, axis=1))


dir_name = "230418-sim-8"

args = np.loadtxt(dir_name + "/args.csv", delimiter=",")
args = args[:, 1]

sim_num = np.loadtxt(dir_name + "/sim_num.csv", dtype=int)
params = np.loadtxt(dir_name + "/params.csv", delimiter=",")

mode = int(params[0])
time_horizon, system_size = int(params[9]), int(params[10])
is_annealed = int(params[11])
num = 100
if mode == 0:
    p_start, p_end, _ = params[1], params[2], int(params[3])
    q, k, m, x0 = params[5], params[6], params[7], params[8]

    plot_fixed_points_p(p_tab=np.linspace(0.0001, 1, num), q=q, k=k, m=m, x0=x0, is_quanched=not is_annealed, is_symmetric=True)
    plt.xlim([p_start, p_end])
    plt.xlabel("p")
    plt.ylabel("a")
elif mode == 1:
    k_start, k_end, _ = params[1], params[2], int(params[3])
    q, p, m, x0 = params[5], params[6], params[7], params[8]

    plot_fixed_points_k(k_tab=np.linspace(k_start, k_end, num), q=q, p=p, m=m, x0=x0, is_quanched=not is_annealed, is_symmetric=True)
    plt.xlim([k_start, k_end])
    plt.xlabel("k")
    plt.ylabel("a")
elif mode == 2:
    m_start, m_end, _ = params[1], params[2], int(params[3])
    q, p, k, x0 = params[5], params[6], params[7], params[8]

    plot_fixed_points_m(m_tab=np.linspace(m_start, m_end, num), q=q, p=p, k=k, x0=x0, is_quanched=not is_annealed,
                        is_symmetric=True)
    plt.xlim([m_start, m_end])
    plt.xlabel("m")
    plt.ylabel("a")
elif mode == 3:
    x0_start, x0_end, _ = params[1], params[2], int(params[3])
    q, p, k, m = params[5], params[6], params[7], params[8]

    plot_fixed_points_x0(x0_tab=np.linspace(x0_start, x0_end, num), q=q, p=p, k=k, m=m, is_quanched=not is_annealed,
                        is_symmetric=True)
    plt.xlim([x0_start, x0_end])
    plt.xlabel("x0")
    plt.ylabel("a")


time_start = 800

form = '<15'
# print(f"{'loaded data:':{form}}{dir_name}",
#       f"{'p_start:':{form}}{p_start}",
#       f"{'p_end:':{form}}{p_end}",
#       f"{'p_num:':{form}}{p_num}",
#       f"{'system_size:':{form}}{system_size}",
#       f"{'is_annealed:':{form}}{is_annealed}",
#       f"{'q:':{form}}{q}",
#       f"{'x0:':{form}}{x0}",
#       f"{'k:':{form}}{k}",
#       f"{'m:':{form}}{m}",
#       f"{'time_horizon:':{form}}{time_horizon}",
#       f"{'time_start:':{form}}{time_start}",
#       f"{'time_average:':{form}}{time_horizon - time_start + 1}",
#       f"{'sim_num:':{form}}{sim_num}",
#       sep="\n")

cs = np.zeros((len(args), sim_num))
xs = np.zeros((len(args), sim_num))
us = np.zeros((len(args), sim_num))

# plt.figure(1)
for i in range(len(args)):
    for sim_number in range(sim_num):
        single_sim = np.loadtxt(dir_name + f"/{i}/sim-{sim_number}.txt")
        single_sim = single_sim[time_start:]
        single_sim2 = [(2 * x - 1) ** 2 for x in single_sim]
        single_sim4 = [(2 * x - 1) ** 4 for x in single_sim]
        # single_sim = [max(x, 1 - x) for x in single_sim]
        cs[i, sim_number] = np.mean(single_sim)
        xs[i, sim_number] = np.mean(single_sim2) - (2 * cs[i, sim_number] - 1) ** 2
        us[i, sim_number] = 1 - np.mean(single_sim4) / (3 * np.mean(single_sim2) ** 2)
        # plt.plot(single_sim)

# plt.figure(2)
# plt.plot(ps, cs, ".-")
# plt.xlabel("nonconformity probability")
# plt.ylabel("concentration")
#

# ps_theo, cs_theo = get_fixed_points(num=100, q=q, f=f, is_quenched=not is_annealed)




plt.errorbar(args, np.mean(cs, axis=1), yerr=get_sem(cs), fmt='gx')

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
