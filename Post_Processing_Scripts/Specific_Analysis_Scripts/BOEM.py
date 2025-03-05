import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('/Users/bagaenzl/BOEM-Oak-Island/Post_Processing_Scripts')
from process_Sig1k import read_raw_h5, remove_low_correlations, transform_beam_ENUD, save_data
#from Post_Processing_Scripts.process_Sig1k import read_raw_h5, remove_low_correlations, transform_beam_ENUD, save_data
#from Post_Processing_Scripts.bulk_stats_Sig1k import bulk_stats_analysis
from bulk_stats_Sig1k import bulk_stats_analysis


def average_vel_plots(path):
    VertVel = pd.read_hdf(os.path.join(path, 'VertVel.h5'))
    EastVel = pd.read_hdf(os.path.join(path, 'EastVel.h5'))
    NorthVel = pd.read_hdf(os.path.join(path, 'NorthVel.h5'))
    ErrVel = pd.read_hdf(os.path.join(path, 'ErrVel.h5'))
    Time = pd.read_hdf(os.path.join(path, 'Time.h5'))

    fig, axs = plt.subplots(4, sharex=True, sharey=True)
    axs[0].plot(
        Time,
        np.nanmean(EastVel, axis=1),
        color="green",
        label="East",
    )
    axs[1].plot(
        Time,
        np.nanmean(NorthVel, axis=1),
        color="red",
        label="North",
    )
    axs[2].plot(
        Time,
        np.nanmean(VertVel, axis=1),
        color="blue",
        label="VertVel1",
    )
    axs[3].plot(
        Time,
        np.nanmean(ErrVel, axis=1),
        color="gray",
        label="Differennce",
    )
    for i in range(len(axs)):
        axs[i].legend()
    fig.suptitle("Velocity Components versus Time")
    fig.supxlabel("Time (DD HH:MM)")
    fig.supylabel("Velocity (m/s)")
    plt.xlim(left=Time.iloc[1], right=Time.iloc[-1])
    plt.show()


# --------- USER INPUT ---------
# directory_path = r"/Volumes/kanarde/BOEM/deployment_1/Raw/S0_103080_hdf/"
# save_dir = r"/Volumes/kanarde/BOEM/deployment_1/Processed/S0_103080/"
#directory_path = r"/Volumes/BOEM/deployment_1//Raw/S0_103080_hdf/"  # Katherine's Macbook
#save_dir = r"/Volumes/BOEM/deployment_1/Processed/S0_103080/"
directory_path = r"/Volumes/kanarde-1/BOEM/deployment_1/Raw/S1_101418_hdf/" # Brooke path
save_dir = r"/Volumes/kanarde-1/BOEM/deployment_1/Processed/S1_101418/" # Brooke path

#directory_path = r'Z:/deployment_1/Raw/S0_103080_hdf/'#Levis path
#save_dir = r'Z:/deployment_1/Processed/'#Levis path

# --------- QUALITY CONTROL ---------
# files = os.listdir(directory_path)  # lists in arbitrary order because there is not a zero in front of folder numbers
# if '.DS_Store' in files:  # remove hidden files on macs
#     files.remove('.DS_Store')
# folder_id = 0  # need to change if starting with group 1

# for file_name in files:
#     # import folder names
#     folder_id += 1
#     print(f"Processing Group {folder_id}")
#     path = os.path.join(directory_path, file_name)
#     print(path)
#     save_path_name = os.path.join(save_dir, f"Group{folder_id}")
#     # Ensure save directory exists
#     #os.makedirs(save_path_name, exist_ok=True)

#     # call post-processing functions
#     Data = read_raw_h5(path)  # KA: needed to install pytables
#     print(f"read in data")
#     # print(Data["CellDepth"])

#     Data = remove_low_correlations(Data)
#     print(f"removed low correlations")

#     Data = transform_beam_ENUD(Data)
#     print("transformed to ENUD")

#     save_data(Data, save_path_name)
#     print(f"Processed {file_name} and saved to {save_dir}")

directory_path=r"/Volumes/kanarde-1/BOEM/deployment_1/Processed/S1_101418/" # Brooke path
savedir=r"/Volumes/kanarde-1/BOEM/deployment_1/BulkStats/S1_101418" # brooke path
# --------- BULK STATISTICS ---------
waves = bulk_stats_analysis(directory_path, save_dir)

# --------- PLOTTING ---------
#average_vel_plots(path)