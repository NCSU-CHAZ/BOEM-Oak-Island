import process_Sig1k as ps
import os

print("hi")
directory_path = r"/Volumes/kanarde-1/BOEM/deployment_1/Raw/S0_103080_hdf/"
save_dir = r"/Volumes/kanarde-1/BOEM/deployment_1/Processed/S0_103080/"
##directory_path = r'Z:/deployment_1/Raw/S0_103080_hdf/'#Levis path
save_dir = r'Z:/deployment_1/Processed/'#Levis path

files=os.listdir(directory_path)
print(files)
i = 0 # need to change if starting with group 1
for file_name in files:
     i+=1
     print(f"Processing Group {i}")
     path = os.path.join(directory_path, file_name)
     print(path)
     save_path_name = os.path.join(save_dir, f"Group{i}")
     #os.makedirs(os.path.dirname(save_path_name), exist_ok=True) # create a new group folder if not already present
     Data=ps.read_raw_h5(path)
     print(f"read in data")
     print(Data["CellDepth"])
     Data=ps.remove_low_correlations(Data)
     print(f"removed low corr")
     Data=ps.transform_beam_ENUD(Data)
     print("transformed to ENUD")
     ps.save_data(Data, save_dir)
     print(f"Processed {file_name} and saved to {save_dir}")

#/Volumes/kanarde/BOEM/deployment_1/Raw/S0_103080_hdf
#/Volumes/kanarde/BOEM/deployment_1/Raw/S0_103080_hdf/Group1/Burst_Beam2xyz.h5
# files = [
#     f
#     for f in os.listdir(directory_path)
#     if os.path.isfile(os.path.join(directory_path, f))
# ]
# i = 0
# print("hi")
# print(files)
# for file_name in files:
#     i += 1
#     print(f"Processing Group {i}")
    
#     print(file_name)
#     path = os.path.join(directory_path, f"Group{i}",file_name)
#     save_path = os.path.join(save_dir, f"Group{i}")
#     Data=ps.read_raw_h5(path)
#     Data=ps.remove_low_correlations(Data)
#     Data=ps.transform_beam_ENUD(Data)
#     ps.save_data(Data, save_dir)
#     print(f"Processed {file_name} and saved to {save_path}")
#     #dtnum_dttime_adcp("/Volumes/kanarde/BOEM/deployment_1/Raw/S0_103080_hdf/Group{:1d}/Burst_Time.h5").format(i)

# Loop through each Group directory
# i = 0
# for group in os.listdir(directory_path):
#     group_path = os.path.join(directory_path, group)
    
#     # Skip .DS_Store and non-directory items
#     if not os.path.isdir(group_path) or group == '.DS_Store':
#         continue

#     i += 1
#     print(f"Processing {group}...")

#     # List files in the Group directory, excluding .DS_Store
#     group_files = [
#         f for f in os.listdir(group_path)
#         if os.path.isfile(os.path.join(group_path, f)) and f != '.DS_Store'
#     ]

#     # Skip if no valid files are found in the Group directory
#     if not group_files:
#         print(f"No valid files found in {group_path}. Skipping Group {i}.")
#         continue

#     # Process each valid file
#     for file_name in group_files:
#         path = os.path.join(group_path)  # Full path to the file
        
#         # Create save path
#         save_group_dir = os.path.join(save_dir, f"Group{i}")
#         if not os.path.exists(save_group_dir):
#             os.makedirs(save_group_dir)
        
#         save_path = os.path.join(save_group_dir)  # Path to save processed file

#         # Read and process data
#         try:
#             Data = ps.read_raw_h5(path)
#             Data = ps.remove_low_correlations(Data)
#             Data = ps.transform_beam_ENUD(Data)
#             ps.save_data(Data, save_path)
#             print(f"Processed {file_name} and saved to {save_path}")
#         except Exception as e:
#             print(f"Error processing {file_name}: {e}")