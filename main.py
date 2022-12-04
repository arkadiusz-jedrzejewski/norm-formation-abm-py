import numpy as np
import os
from datetime import date

if __name__ == "__main__":

    today = date.today()
    print(f"Today is {today}")
    print(type(today.strftime("%Y-%m-%d")))
    print(today.strftime("%y%m%d"))

    print(os.listdir())
    dirlist = [item for item in os.listdir() if os.path.isdir(item)]
    print(dirlist)
    res = []
    for item in dirlist:
        print(item)
        print(item.split("-"))
        items = item.split("-")
        if len(items) == 3 and items[1] == "sim":
            if today.strftime("%y%m%d") == items[0]:
                print("ll")
                res.append(items[2])

    next_index = 1
    if len(res) != 0:
        next_index = int(max(res)) + 1
    print(next_index)
    dir_name = today.strftime("%y%m%d") + '-sim-' + str(next_index)
    print(dir_name)
    os.mkdir(dir_name)


    # os.system("norm_formation_abm.exe 10 0.2 3 0.5 10000 1 200 name.txt")
