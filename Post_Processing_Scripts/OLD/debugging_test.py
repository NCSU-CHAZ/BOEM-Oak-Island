import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
import re

dirpath = r'Z:/deployment_1/Processed/S0_103080/'

group_dirs = [entry for entry in os.scandir(dirpath) if entry.is_dir() and entry.name.startswith('Group')]
Nsamp = 3600*4
# Sort the directories to ensure you process them in order
group_dirs.sort(key=lambda x: int(x.name.replace('Group', '')))
Waves = {"Time": pd.DataFrame([])}
gg=0
for group_dir in group_dirs:
    group_path = group_dir.path  # Get the full path of the current group
    Time = pd.read_hdf(os.path.join(r"Z:\deployment_1\Processed\S0_103080\Group1", "Time.h5"))

    nt = len(Time)
    N = math.floor(nt / Nsamp)
 
    for i in range(N):
        t = Time.iloc[i * Nsamp: Nsamp * (i + 1)]

        tavg = t.iloc[
            round(Nsamp / 2)
        ]  # Take the time for this ensemble by grabbing the middle time

        Waves["Time"] = pd.concat(
            [Waves["Time"], pd.DataFrame([tavg])], ignore_index=True
        )
    break

directory_path_mat = r'Z:/deployment_1/Raw/S1_101418_mat/'

files = [
        f
        for f in os.listdir(directory_path_mat)
        if os.path.isfile(os.path.join(directory_path_mat, f))
    ]

files.sort(key=lambda x: int(re.search(r"NCSU_(\d+)", x).group(1)) if re.search(r"NCSU_(\d+)", x) else float('inf'))



i = 0
    
for file_name in files:
    print(file_name)
    i += 1
    path = os.path.join(directory_path_mat, file_name)
    save_path = os.path.join(r'Z:/deployment_1/Raw/S1_101418_mat/', f"Group{i}")
    print(save_path)

