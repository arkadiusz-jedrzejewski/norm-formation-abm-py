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
    for item in dirlist:
        print(item)
        print(item.split("-"))


    # os.system("norm_formation_abm.exe 10 0.2 3 0.5 10000 1 200 name.txt")
