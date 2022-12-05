import numpy as np
import os
from datetime import date


def get_dir_name():
    today = date.today()
    dirlist = [item for item in os.listdir() if os.path.isdir(item)]
    res = []

    for item in dirlist:
        items = item.split("-")
        if len(items) == 3 and items[1] == "sim":
            if today.strftime("%y%m%d") == items[0]:
                res.append(items[2])

    next_index = 1
    if len(res) != 0:
        next_index = int(max(res)) + 1

    dir_name = today.strftime("%y%m%d") + '-sim-' + str(next_index)

    return dir_name


if __name__ == "__main__":
    dir_name = get_dir_name()
    print(dir_name)
    os.mkdir(dir_name)

    # os.system("norm_formation_abm.exe 10 0.2 3 0.5 10000 1 200 name.txt")
