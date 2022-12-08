import numpy as np
import os
from datetime import date
import subprocess
import multiprocessing as mp
from functools import partial


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


def run_single_sim(p_tuple, q, f, system_size, init_opinions, time_horizon, p_dir_name):
    p_index, p, sim_number, seed = p_tuple
    file_name = p_dir_name + f"/{p_index}/sim-{sim_number}.txt"
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

    sim_num = 2
    sims = np.arange(sim_num)

    q = 3
    f = 0.5
    system_size = 10000
    time_horizon = 1000

    print(ps)
    print(ps_index)
    np.savetxt(dir_name + "/probs.csv", np.column_stack((ps_index, ps)), fmt="%i, %.18f")
    np.savetxt(dir_name + "/sim_num.csv", [sim_num], fmt="%i")
    np.savetxt(dir_name + "/params.csv",
               [[p_start, p_stop, p_num, sim_num, q, f, time_horizon, system_size]],
               fmt="%.8f, " * 2 + "%i, %i, %.8f, %.8f, %i, %i",
               header="p_start, p_stop, p_num, sim_num, q, f, time_horizon, system_size")

    p_tuples = []
    seed = 1
    for p_index in ps_index:
        if not os.path.exists(dir_name + f"/{p_index}"):
            os.mkdir(dir_name + f"/{p_index}")
        seeds = []
        for sim_number in sims:
            p_tuples.append((p_index, ps[p_index], sim_number, seed))
            seeds.append(seed)
            seed += 1000
        np.savetxt(dir_name + f"/{p_index}/seeds.csv", np.column_stack((sims, seeds)), fmt="%i, %i")

    with mp.Pool(6) as pool:
        pool.map(partial(run_single_sim,
                         q=q,
                         f=f,
                         system_size=system_size,
                         init_opinions=1,
                         time_horizon=time_horizon,
                         p_dir_name=dir_name
                         ),
                 p_tuples)

    # os.system("norm_formation_abm.exe 10 0.2 3 0.5 10000 1 200 name.txt")
