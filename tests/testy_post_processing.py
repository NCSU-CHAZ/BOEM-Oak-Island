import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import assert_array_almost_equal
import os
from Post_Processing_Scripts.process_Sig1k import read_Sig1k, read_raw_h5, remove_low_correlations, \
    transform_beam_ENUD, save_data
from Post_Processing_Scripts.bulk_stats_Sig1k import bulk_stats_analysis
import re

deployment_num = 1
sensor_id = "S0_103080"  # S1_101418
directory_initial_user_path = r"/Volumes/BOEM/"  # Katherine

# define which processing steps you would like to perform
run_convert_mat_h5 = True
run_quality_control = True
compare_velocities = True

group_id = 7  # specify which group you want to compare

###############################################################################
# create paths to save directories
###############################################################################
directory_path_mat = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/Raw/", sensor_id+"_mat/")
save_dir_raw = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/Raw/", sensor_id+"_hdf/")
save_dir_qc = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/Processed/", sensor_id+"/")
save_dir_bulk_stats = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/BulkStats/", sensor_id+"/")

###############################################################################
# convert mat files to h5 files
###############################################################################
if run_convert_mat_h5:

    files = [
        f
        for f in os.listdir(directory_path_mat)
        if os.path.isfile(os.path.join(directory_path_mat, f))
    ]
    files.sort(key=lambda x: int(re.search(r"NCSU_(\d+)", x).group(1)) if re.search(r"NCSU_(\d+)", x) else float('inf'))

    file_id = group_id - 1
    path = os.path.join(directory_path_mat, files[file_id])
    file_id += 1
    if file_id < 10:
        save_path = os.path.join(save_dir_raw, f"Group0{file_id}")
    else:
        save_path = os.path.join(save_dir_raw, f"Group{file_id}")
    read_Sig1k(path, save_path)

###############################################################################
# quality control
###############################################################################
if run_quality_control:

    files = sorted(os.listdir(save_dir_raw))

    if '.DS_Store' in files:  # remove hidden files on macs
        files.remove('.DS_Store')
    folder_id = group_id - 1

    path = os.path.join(save_dir_raw, files[folder_id])
    folder_id += 1
    print(path)
    if folder_id < 10:
        save_path_name = os.path.join(save_dir_qc, f"Group0{folder_id}")
    else:
        save_path_name = os.path.join(save_dir_qc, f"Group{folder_id}")
    print(f"Processing {files[folder_id-1]}")  # for debugging

    # call post-processing functions
    Data = read_raw_h5(path)  # KA: needed to install pytables
    print(f"read in data")

    Data = remove_low_correlations(Data)
    print(f"removed low correlations")

    Data = transform_beam_ENUD(Data)
    print("transformed to ENUD")

    save_data(Data, save_path_name)
    print(f"Processed {files[folder_id-1]} and saved to {save_dir_qc}")

# plot velocities
if compare_velocities:
    # group_dirs = [entry for entry in os.scandir(save_dir_qc) if entry.is_dir() and entry.name.startswith('Group')]
    #
    # # Sort the directories to ensure you process them in order
    # group_dirs.sort(key=lambda x: int(x.name.replace('Group', '')))
    #
    # folder_id = group_id - 1
    #
    # # Start loop that will load in data for each variable from each day ("group")
    # group_path = group_dirs[folder_id].path  # Get the full path of the current group
    # VertVel = pd.read_hdf(os.path.join(group_path, "VertVel.h5"))
    # EastVel = pd.read_hdf(os.path.join(group_path, "EastVel.h5"))
    # NorthVel = pd.read_hdf(os.path.join(group_path, "NorthVel.h5"))

    # fig, axs = plt.subplots(1, 2, figsize=(10, 3))
    # axs[0].plot(np.array(Data["EastVel"].iloc[0:1000, 1]))  # only plot the second bin
    # axs[0].plot(np.array(Data["VelE_mat"].iloc[0:1000, 1]))

    # Take the difference of our beam-derived (and QC'ed) water levels and the original ENU from Matlab (not QC'ed)
    np.max(Data["EastVel"] - Data["VelE_mat"])  # 0.01 to 0.3 m/s
    np.max(Data["NorthVel"] - Data["VelN_mat"])  # ~0.03 m/s
    np.max(Data["VertVel"] - Data["VelU_mat"])  # very small, e^7 m/s
    np.max(Data["ErrVel"] - Data["VelDiff_mat"])  # these are substantially different -- not sure why, ~2-3 m/s