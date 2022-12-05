import numpy as np
import os
from datetime import date
import subprocess
import multiprocessing as mp


def get_dir_name():
    today = date.today()
    dirlist = [item for item in os.listdir() if os.path.isdir(item)]
    res = []

    for item in dirlist:
        items = item.split("-")
        if len(items) == 3 and items[1] == "sim":
            if today.strftime("%y%m%d") == items[0]:
                res.append(int(items[2]))

    next_index = 1
    if len(res) != 0:
        next_index = max(res) + 1

    dir_name = today.strftime("%y%m%d") + '-sim-' + str(next_index)

    return dir_name


def run_single_sim(seed, p_tuple, q, f, system_size, init_opinions, time_horizon, p_dir_name):
    p_index, p, sim_number = p_tuple
    file_name = p_dir_name + f"/{p_index}/sim-{sim_number}"
    subprocess.run(
        f"norm_formation_abm.exe {seed} {p} {q} {f} {system_size} {init_opinions} {time_horizon} {file_name}")


if __name__ == "__main__":
    dir_name = get_dir_name()
    os.mkdir(dir_name)

    # parameters
    p_start = 0
    p_stop = 1
    p_num = 20
    ps = np.linspace(p_start, p_stop, p_num)
    ps_index = np.arange(p_num)

    sim_num = 10
    sims = np.arange(sim_num)

    print(ps)
    print(ps_index)
    np.savetxt(dir_name + "/probs.csv", np.column_stack((ps_index, ps)), fmt=("%i", "%.18f"))
    np.savetxt(dir_name + "/sim_num.csv", [sim_num], fmt="%i")
    np.savetxt(dir_name + "/params.csv",
               [[p_start, p_stop, p_num, sim_num]],
               fmt="%.8f, " * 2 + "%i, %i",
               header="p_start, p_stop, p_num, sim_num")

    for i in ps_index:
        if not os.path.exists(dir_name + f"/{i}"):
            os.mkdir(dir_name + f"/{i}")

    run_single_sim(seed=10,
                   p_tuple=(0, 0.2, 1),
                   q=3,
                   f=0.5,
                   system_size=10000,
                   init_opinions=1,
                   time_horizon=200,
                   p_dir_name=dir_name)
    # os.system("norm_formation_abm.exe 10 0.2 3 0.5 10000 1 200 name.txt")
