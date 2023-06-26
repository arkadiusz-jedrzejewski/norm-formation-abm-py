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


def run_single_sim(arg0_tuple, is_annealed, is_symmetric, q, arg1, arg2, arg3, system_size, init_opinions, time_horizon, p_dir_name,
                   mode):
    if mode == 0:
        arg0_index, p, sim_number, seed = arg0_tuple
        k = arg1
        m = arg2
        x0 = arg3
    elif mode == 1:
        arg0_index, k, sim_number, seed = arg0_tuple
        p = arg1
        m = arg2
        x0 = arg3
    elif mode == 2:
        arg0_index, m, sim_number, seed = arg0_tuple
        p = arg1
        k = arg2
        x0 = arg3
    elif mode == 3:
        arg0_index, x0, sim_number, seed = arg0_tuple
        p = arg1
        k = arg2
        m = arg3

    file_name = p_dir_name + f"/{arg0_index}/sim-{sim_number}.txt"
    is_annealed = 1 if is_annealed else 0
    is_symmetric = 1 if is_symmetric else 0
    subprocess.run(
        f"norm_formation_abm.exe {file_name} {seed} {system_size} {time_horizon} {is_annealed} {is_symmetric} {init_opinions} {q} {x0} {k} {m} {p}")


def create_diagram(p, is_annealed, is_symmetric, sim_num, q, x0, k, m, system_size, time_horizon,
                   seed):
    if type(p) is tuple:
        arg_start, arg_stop, arg_num = p
        header = "mode, p_start, p_stop, p_num, sim_num, q, k, m, x0, time_horizon, system_size, is_annealed, is_symmetric"
        arg1 = k
        arg2 = m
        arg3 = x0
        mode = 0
    elif type(k) is tuple:
        arg_start, arg_stop, arg_num = k
        header = "mode, k_start, k_stop, k_num, sim_num, q, p, m, x0, time_horizon, system_size, is_annealed, is_symmetric"
        arg1 = p
        arg2 = m
        arg3 = x0
        mode = 1
    elif type(m) is tuple:
        arg_start, arg_stop, arg_num = m
        header = "mode, m_start, m_stop, m_num, sim_num, q, p, k, x0, time_horizon, system_size, is_annealed, is_symmetric"
        arg1 = p
        arg2 = k
        arg3 = x0
        mode = 2
    elif type(x0) is tuple:
        arg_start, arg_stop, arg_num = x0
        header = "mode, x0_start, x0_stop, x0_num, sim_num, q, p, k, m, time_horizon, system_size, is_annealed, is_symmetric"
        arg1 = p
        arg2 = k
        arg3 = m
        mode = 3

    args = np.linspace(arg_start, arg_stop, arg_num)
    args_index = np.arange(arg_num)

    sims = np.arange(sim_num)

    np.savetxt(dir_name + "/args.csv", np.column_stack((args_index, args)), fmt="%i, %.18f")
    np.savetxt(dir_name + "/sim_num.csv", [sim_num], fmt="%i")
    np.savetxt(dir_name + "/params.csv",
               [[mode, arg_start, arg_stop, arg_num, sim_num, q, arg1, arg2, arg3, time_horizon, system_size,
                 is_annealed, is_symmetric]],
               fmt="%i, " + "%.8f, " * 2 + "%i, %i, %.8f, %.8f, %.8f, %.8f, %i, %i, %i, %i",
               header=header)

    arg0_tuples = []

    for arg_index in args_index:
        if not os.path.exists(dir_name + f"/{arg_index}"):
            os.mkdir(dir_name + f"/{arg_index}")
        seeds = []
        for sim_number in sims:
            arg0_tuples.append((arg_index, args[arg_index], sim_number, seed))
            seeds.append(seed)
            seed += 1000
        np.savetxt(dir_name + f"/{arg_index}/seeds.csv", np.column_stack((sims, seeds)), fmt="%i, %i")

    with mp.Pool(6) as pool:
        pool.map(partial(run_single_sim,
                         is_annealed=is_annealed,
                         is_symmetric=is_symmetric,
                         q=q,
                         arg1=arg1,
                         arg2=arg2,
                         arg3=arg3,
                         system_size=system_size,
                         init_opinions=1,
                         time_horizon=time_horizon,
                         p_dir_name=dir_name,
                         mode=mode
                         ),
                 arg0_tuples)


if __name__ == "__main__":
    dir_name = get_dir_name()
    os.mkdir(dir_name)

    # create_diagram(p=0.3,
    #                is_annealed=True,
    #                sim_num=2,
    #                q=3,
    #                x0=0.5,
    #                k=(-50, 20, 25),
    #                m=0.6,
    #                system_size=10000,
    #                time_horizon=1000,
    #                seed=1)
    # create_diagram(p=(0, 1, 25),
    #                is_annealed=True,
    #                sim_num=2,
    #                q=3,
    #                x0=0.5,
    #                k=-10,
    #                m=0.6,
    #                system_size=10000,
    #                time_horizon=1000,
    #                seed=1)
    create_diagram(p=(0, 1, 11),
                   is_annealed=False,
                   is_symmetric=True,
                   sim_num=10000,
                   q=3,
                   x0=0.5,
                   k=-15,
                   m=0.5,
                   system_size=10,
                   time_horizon=5,
                   seed=1)
