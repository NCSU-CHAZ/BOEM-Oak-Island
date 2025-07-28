import os
import numpy as np
import re
from analysis_bulkstats import (
    load_qc_data,
    sediment_analysis,
    save_waves,
    bulk_stats_depth_averages,
    initialize_bulk, calculate_wave_stats
)
from analysis import (
    read_Sig1k,
    read_data_h5,
    remove_low_correlations,
    transform_beam_ENUD,
    save_data,
)

###############################################################################
# user input
###############################################################################

deployment_num = 1
sensor_id = "E1_103071"  # S1_101418 or S0_103080
# directory_initial_user_path = r"/Volumes/BOEM/"  # Katherine
# directory_initial_user_path = r"/Volumes/kanarde/BOEM/"  # Brooke /
directory_initial_user_path = r"Z:/"  # Levi

# define which processing steps you would like to perform
run_convert_mat_h5 = False
run_quality_control = False
run_bulk_statistics = True
echosounder = (
    True  # set to True if you want to process echosounder data, False for vertical beam
)


group_id = 1  # specify if you want to process starting at a specific group_id; must be 1 or greater
group_ids_exclude = [
    0
]  # for processing bulk statistics; skip group 1 (need to add a line of code in bulk stats to
# remove 1 so that I can make [1,2] here

###############################################################################
# create paths to save directories
###############################################################################
directory_path_mat = os.path.join(
    directory_initial_user_path,
    f"deployment_{deployment_num}/Raw/",
    sensor_id + "_mat/",
)
save_dir_data = os.path.join(
    directory_initial_user_path,
    f"deployment_{deployment_num}/Raw/",
    sensor_id + "_hdf/",
)
save_dir_qc = os.path.join(
    directory_initial_user_path,
    f"deployment_{deployment_num}/Processed/",
    sensor_id + "/",
)
save_dir_bulk_stats = os.path.join(
    directory_initial_user_path,
    f"deployment_{deployment_num}/BulkStats/",
    sensor_id + "/",
)
sbepath = os.path.join(
    directory_initial_user_path,
    f"deployment_{deployment_num}/Raw/",
    "E1RBR",
    "SBE_00003570_DEP4_FPSE1_L0.mat",
)
config_path = os.path.join(
    directory_initial_user_path,
    f"deployment_{deployment_num}/Raw/",
    sensor_id + "_mat/" + "SIG_00103071_DEP4_FPSE1_config.mat",
)

###############################################################################
# convert mat files to h5 files
###############################################################################
if run_convert_mat_h5:
    print("Running mat conversion")
    files = [
        f
        for f in os.listdir(directory_path_mat)
        if os.path.isfile(os.path.join(directory_path_mat, f))
    ]

    files.sort(
        key=lambda x: (
            int(re.search(r"FPS4_(\d+)", x).group(1))
            if re.search(r"FPS4_(\d+)", x)
            else float("inf")
        )
    )

    file_id = group_id - 1

    for file_name in files[file_id:]:
        file_id += 1
        path = os.path.join(directory_path_mat, file_name)
        print(path)
        if file_id < 10:
            save_path = os.path.join(save_dir_data, f"Group0{file_id}")
        else:
            save_path = os.path.join(save_dir_data, f"Group{file_id}")
        # read_Sig1k(path, config_path ,save_path) #Why is here, this makes it read twice?
        try:
            read_Sig1k(path, config_path, save_path)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")


###############################################################################
# quality control
###############################################################################
if run_quality_control:
    print("Running Quality Control")

    files = sorted(os.listdir(save_dir_data))

    if ".DS_Store" in files:  # remove hidden files on macs
        files.remove(".DS_Store")
    folder_id = group_id - 1

    for file_name in files[folder_id:]:
        # import folder names
        folder_id += 1
        path = os.path.join(save_dir_data, file_name)
        print(path)
        if folder_id < 10:
            save_path_name = os.path.join(save_dir_qc, f"Group0{folder_id}")
        else:
            save_path_name = os.path.join(save_dir_qc, f"Group{folder_id}")
        print(f"Processing {file_name}")  # for debugging
        try:
            # call post-processing functions
            Data = read_data_h5(path)  # KA: needed to install pytables
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

# waves = bulk_stats_analysis(save_dir_qc, save_dir_bulk_stats, group_ids_exclude,sbepath)


if run_bulk_statistics:
    try:
        print("Running bulk statistics")

        Waves, sbe = initialize_bulk(
            save_dir_qc,
            sbepath,
            dtburst=3600,
            dtens=512,
            fs=4,
            sensor_height=0.508,
            depth_threshold=3,
        )

        group_dirs = [
            entry
            for entry in os.scandir(save_dir_qc)
            if entry.is_dir() and entry.name.startswith("Group")
        ]

        # Sort the directories to ensure you process them in order
        group_dirs.sort(key=lambda x: int(x.name.replace("Group", "")))

        # only loop through directories specified by the user
        for index in sorted(group_ids_exclude, reverse=True):
            del group_dirs[index]

        for group_dir in group_dirs:
            group_path = group_dir.path  # Get the full path of the current group
            Data, Waves = load_qc_data(group_path, Waves)
            if echosounder:
                print("analysing echosounder data")
                Data, Waves = sediment_analysis(Waves, Data, sbe, 0.330)
            else:
                print("analyising vertical beam")
                # Data, Waves = sediment_analysis_vert(
                #     Data, Waves, sbe, 0.330, vertical_beam=True
                # )

            dtburst = 3600  # duration of each burst in seconds
            fs = 4  # sampling frequency
            # Get number of total samples in group
            nt = len(Data["Time"])
            Nsamp = dtburst * fs  # number of samples per burst
            N = nt // Nsamp
            Nb = len(Data["Celldepth"])  # Number of bins
            
            print("Iterating...")

            # Loop over ensembles ("bursts")
            for i in range(N):
            
                Waves, Data = bulk_stats_depth_averages(
                    Waves,
                    Data,
                    i,
                    Nsamp,
                    sensor_height=0.508,
                    dtburst=dtburst,
                    dtens=512,
                    fs=fs,
                )

                Waves = calculate_wave_stats(
                    Waves, Data, Nsamp, i, 
                    sensor_height=0.508, fs=4, dtburst=3600, dtens=512)
                
            print(f"Processed {group_path} for bulk statistics")

            # Save the processed waves data

        save_waves(Waves, group_path)
    
    except Exception as e:
        print(f"Error processing {e}")

