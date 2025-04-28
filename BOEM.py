"""Run script for BOEM deployments: process and analyze data from Nortek Signature 1000s

Here we provide a template for the workflow used in quality control and statistical analysis of bottom mounted ADCP data
as part of the lander deployments at Frying Pan Shoals (on behalf of BOEM and Oak Island).

References
----------
None

Notes
---------
None

"""

import os
from Post_Processing_Scripts.process_Sig1k import read_Sig1k, read_raw_h5, remove_low_correlations, \
    transform_beam_ENUD, save_data
from Post_Processing_Scripts.bulk_stats_Sig1k import bulk_stats_analysis
import re
import itertools

###############################################################################
# user input
###############################################################################

deployment_num = 1
sensor_id = "S1_101418"  # S1_101418 #S0_103080
directory_initial_user_path = r"/Volumes/BOEM/"  # Katherine
# directory_initial_user_path = r"/Volumes/kanarde/BOEM/"  # Brooke
# directory_initial_user_path = r"Z:/"  # Levi

# define which processing steps you would like to perform
run_convert_mat_h5 = False
run_quality_control = False
run_bulk_statistics = True

group_id = 1  # specify if you want to process starting at a specific group_id; must be 1 or greater
group_ids_exclude = [0]  # for processing bulk statistics; skip group 1 (need to add a line of code in bulk stats to
# remove 1 so that I can make [1,2] here

###############################################################################
# create paths to save directories
###############################################################################
directory_path_mat = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/Raw/", sensor_id + "_mat/")
save_dir_raw = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/Raw/", sensor_id + "_hdf/")
save_dir_qc = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/Processed/", sensor_id + "/")
save_dir_bulk_stats = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/BulkStats/",
                                   sensor_id + "/")

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

    for file_name in files[file_id:]:
        file_id += 1
        path = os.path.join(directory_path_mat, file_name)
        print(path)
        if file_id < 10:
            save_path = os.path.join(save_dir_raw, f"Group0{file_id}")
        else:
            save_path = os.path.join(save_dir_raw, f"Group{file_id}")
        # read_Sig1k(path, save_path)
        try:
            read_Sig1k(path, save_path)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

###############################################################################
# quality control
###############################################################################
if run_quality_control:

    files = sorted(os.listdir(save_dir_raw))

    if '.DS_Store' in files:  # remove hidden files on macs
        files.remove('.DS_Store')
    folder_id = group_id - 1

    for file_name in files[folder_id:]:
        # import folder names
        folder_id += 1
        path = os.path.join(save_dir_raw, file_name)
        print(path)
        if folder_id < 10:
            save_path_name = os.path.join(save_dir_qc, f"Group0{folder_id}")
        else:
            save_path_name = os.path.join(save_dir_qc, f"Group{folder_id}")
        print(f"Processing {file_name}")  # for debugging
        try:
            # call post-processing functions
            Data = read_raw_h5(path)  # KA: needed to install pytables
            print(f"read in data")

            Data = remove_low_correlations(Data)
            print(f"removed low correlations")

            Data = transform_beam_ENUD(Data)
            print("transformed to ENUD")

            save_data(Data, save_path_name)
            print(f"Processed {file_name} and saved to {save_dir_qc}")
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

###############################################################################
# bulk statistics
###############################################################################
if run_bulk_statistics:
    waves = bulk_stats_analysis(save_dir_qc, save_dir_bulk_stats, group_ids_exclude)
