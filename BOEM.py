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

###############################################################################
# user input
###############################################################################

# define paths to raw data and save directories
directory_path_mat = r"/Volumes/kanarde/BOEM/deployment_1/Raw/S1_101418_mat/"  # Katherine's paths
save_dir_raw = r"/Volumes/kanarde/BOEM/deployment_1/Raw/S1_101418_hdf/" 
save_dir_qc = r"/Volumes/kanarde/BOEM/deployment_1/Processed/S1_101418/"
save_dir_bulk_stats = r"/Volumes/kanarde/BOEM/deployment_1/BulkStats/S1_101418"
# save_dir_raw = r"/Volumes/kanarde/BOEM/deployment_1/Raw/S1_101418_hdf/" # Brooke's paths
# save_dir_qc = r"/Volumes/kanarde/BOEM/deployment_1/Processed/S1_101418/"
# save_dir_bulk_stats = r"/Volumes/kanarde/BOEM/deployment_1/BulkStats/S1_101418"
# save_dir_raw = r'Z:/deployment_1/Raw/S0_103080_hdf/'  # Levi's paths
# save_dir_qc = r'Z:/deployment_1/Processed/'
# directory_path_mat = r"Z:\deployment_1\Raw\S0_103080_mat"

# define which processing steps you would like to perform
run_convert_mat_h5 = False
run_quality_control = True
run_bulk_statistics = False

###############################################################################
# convert mat files to h5 files
###############################################################################
if run_convert_mat_h5:
    
    files = [
        f
        for f in os.listdir(directory_path_mat)
        if os.path.isfile(os.path.join(directory_path_mat, f))
    ]
    i = 0
    
    for file_name in files:
        i += 1
        path = os.path.join(directory_path_mat, file_name)
        save_path = os.path.join(save_dir_raw, f"Group{i}")
        read_Sig1k(path, save_path)

###############################################################################
# quality control
###############################################################################
if run_quality_control:

    files = os.listdir(save_dir_raw)  # lists in arbitrary order because there is not a zero in front of folder #s

    if '.DS_Store' in files:  # remove hidden files on macs
        files.remove('.DS_Store')
    folder_id = 0  # need to change if starting with group 1

    for file_name in files:
        # import folder names
        folder_id += 1
        print(f"Processing Group {folder_id}")
        path = os.path.join(save_dir_raw, file_name)
        print(path)
        save_path_name = os.path.join(save_dir_qc, f"Group{folder_id}")

        # call post-processing functions
        Data = read_raw_h5(path)  # KA: needed to install pytables
        print(f"read in data")

        Data = remove_low_correlations(Data)
        print(f"removed low correlations")

        Data = transform_beam_ENUD(Data)
        print("transformed to ENUD")

        save_data(Data, save_path_name)
        print(f"Processed {file_name} and saved to {save_dir_qc}")

###############################################################################
# bulk statistics
###############################################################################
if run_bulk_statistics:

    waves = bulk_stats_analysis(save_dir_qc, save_dir_bulk_stats)