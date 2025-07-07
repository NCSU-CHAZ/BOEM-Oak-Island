from scipy.io import loadmat
import pandas as pd
import os
import h5py
import numpy as np
import re
from datetime import datetime, timedelta
from analysis_bulkstats import bulk_stats_analysis

###############################################################################
# user input
###############################################################################

deployment_num = 1
sensor_id = "E1_103071"  # S1_101418 or S0_103080
#directory_initial_user_path = r"/Volumes/BOEM/"  # Katherine
# directory_initial_user_path = r"/Volumes/kanarde/BOEM/"  # Brooke /
directory_initial_user_path = r"Z:/"  # Levi

# define which processing steps you would like to perform
run_convert_mat_h5 = False
run_quality_control = True
run_bulk_statistics = True


group_id = 1 # specify if you want to process starting at a specific group_id; must be 1 or greater
group_ids_exclude = [0]  # for processing bulk statistics; skip group 1 (need to add a line of code in bulk stats to
# remove 1 so that I can make [1,2] here

###############################################################################
# create paths to save directories
###############################################################################
directory_path_mat = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/Raw/", sensor_id + "_mat/")
save_dir_data = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/Raw/", sensor_id + "_hdf/")
save_dir_qc = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/Processed/", sensor_id + "/")
save_dir_bulk_stats = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/BulkStats/",
                                   sensor_id + "/")
config_path = os.path.join(directory_initial_user_path, f"deployment_{deployment_num}/Raw/", sensor_id + "_mat/" + "SIG_00103071_DEP4_FPSE1_config.mat")

""""""""""""""""""
##################
""""""""""""""""""


def dtnum_dttime_adcp(datenum_array):
    dates = []
    for datenum in datenum_array:
        python_datetime = datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)
        dates.append(python_datetime)
    return dates

def read_Sig1k(filepath,config_filepath, save_dir):  # Create read function
    Data = loadmat(
        filepath
    )  # Load mat oragnizes the 4 different data structures of the .mat file (Units, Config, Data, Description) as a
    # dictionary with four nested numpy arrays with dtypes as data field titles
    ADCPData = {}  # Initialize the dictionary we'll use
    Config = Data['Config'][0,0]
    # Save the correlation data matrix
    # Save BEAM coordinate velocity matrix
    VelArray = Data["Data"][0, 0]["Burst_Velocity_Beam"]
    reshaped = VelArray.reshape(VelArray.shape[0], -1)
    del VelArray
    ADCPData["Burst_VelBeam"] = pd.DataFrame(reshaped)

    # Save Vertical Amplitude coordinate velocity matrix
    VelArray = Data["Data"][0, 0]["IBurst_Amplitude_Beam"]
    reshaped = VelArray.reshape(VelArray.shape[0],-1)
    del VelArray
    ADCPData['Burst_VertAmplitude'] = pd.DataFrame(reshaped)

    # Save the correlation data matrix
    CorArray = Data["Data"][0, 0]["Burst_Correlation_Beam"]
    reshaped = CorArray.reshape(CorArray.shape[0], -1)
    ADCPData["Burst_CorBeam"] = pd.DataFrame(reshaped)
    del CorArray, reshaped

    # Save the slanted amplitude data matrix
    CorArray = Data["Data"][0, 0]["Burst_Amplitude_Beam"]
    reshaped = CorArray.reshape(CorArray.shape[0], -1)
    ADCPData["Burst_AmpBeam"] = pd.DataFrame(reshaped)
    del CorArray, reshaped

    # Save other fields
    ADCPData["Burst_Time"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Time"])
    ADCPData["Burst_NCells"] = pd.DataFrame(Data["Data"][0, 0]["Burst_NCells"])
    ADCPData["Burst_Pressure"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Pressure"])
    ADCPData["Burst_Heading"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Heading"])
    ADCPData["Burst_Pitch"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Pitch"])
    ADCPData["Burst_Roll"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Roll"])
    ADCPData["Burst_Pitch"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Pitch"])
    ADCPData["Echo1"] = pd.DataFrame(Data["Data"][0, 0]["Echo1Bin1_1000kHz_Echo"])
    ADCPData["Echo2"] = pd.DataFrame(Data["Data"][0, 0]["Echo2Bin1_1000kHz_Echo"])
    # Make directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save config parameters that are needed for processing
    CellSize = pd.DataFrame(Config["Burst_CellSize"])
    SampleRate = pd.DataFrame(Config["Burst_SamplingRate"])
    Beam2xyz = pd.DataFrame(Config["Burst_Beam2xyz"])
    BlankDist = pd.DataFrame(Config["EchoSounder_BlankingDistance"])
    EchoCellSize = pd.DataFrame(Config["EchoSounder_CellSize"])
    EchoFr = pd.DataFrame(Config["EchoSounder_Frequency1"])
    EchoTransmitLength = pd.DataFrame(Config["EchoSounder_TransmitLength1"])
    Beam2xyz = pd.DataFrame(Config["Burst_Beam2xyz"])
    SampleRate = pd.DataFrame(Config["Burst_SamplingRate"])
    NCells = pd.DataFrame(Config["Burst_NCells"])
    
    # Save files there
    NCells.to_hdf(os.path.join(save_dir, "Burst_NCells.h5"), key="df", mode="w")
    CellSize.to_hdf(os.path.join(save_dir, "Burst_CellSize.h5"), key="df", mode="w")
    BlankDist.to_hdf(
        os.path.join(save_dir, "EchoBlankingDistance.h5"), key="df", mode="w"
    )
    EchoCellSize.to_hdf(os.path.join(save_dir, "EchoCellSize.h5"), key="df", mode="w")
    EchoFr.to_hdf(
        os.path.join(save_dir, "EchoFrequency.h5"), key="df", mode="w"
    )
    Beam2xyz.to_hdf(os.path.join(save_dir, "Beam2xyz.h5"), key="df", mode="w")
    EchoTransmitLength.to_hdf(os.path.join(save_dir, "EchoTransmitLength.h5"), key="df", mode="w")
    SampleRate.to_hdf(os.path.join(save_dir, "Burst_SampleRate.h5"), key="df", mode="w")
    
    for field_name, df in ADCPData.items():
        save_path = os.path.join(save_dir, f"{field_name}.h5")
        df.to_hdf(save_path, key="df", mode="w")
        print(f"Saved {field_name} to {save_path}")
    
    print(f"Converted mat to hdr for {save_path}")

def read_data_h5(path):
    """
    Read h5 files of data data from Sig1000. Raw data converted from mat to h5 in 'read_Sig1k'

    :param:
    path: string
        path to directory to data h5 files
    save_dir: string
        path to directory where processed, quality controlled data should be saved

    :return:
    Data: dictionary
        data data as a dictionary
    """

    # initialize the Data dictionary as well as it's keys
    Data = {}

    Data['CorBeam'] = pd.read_hdf(os.path.join(path, 'Burst_CorBeam.h5'))
    Data['AmpBeam'] = pd.read_hdf(os.path.join(path, 'Burst_AmpBeam.h5'))
    Data['Heading'] = pd.read_hdf(os.path.join(path, 'Burst_Heading.h5'))
    Data['Pressure'] = pd.read_hdf(os.path.join(path, 'Burst_Pressure.h5'))
    Data['Roll'] = pd.read_hdf(os.path.join(path, 'Burst_Roll.h5'))
    datenum_array = pd.read_hdf(os.path.join(path, 'Burst_Time.h5'))
    Data['VelBeam'] = pd.read_hdf(os.path.join(path, 'Burst_VelBeam.h5'))
    Data['TransmitLength'] = pd.read_hdf(os.path.join(path, 'EchoTransmitLength.h5'))
    Data['Pitch'] = pd.read_hdf(os.path.join(path, 'Burst_Pitch.h5'))
    Data['Echo1'] = pd.read_hdf(os.path.join(path, 'Echo1.h5'))
    Data['Echo2'] = pd.read_hdf(os.path.join(path, 'Echo2.h5'))
    Data['Beam2xyz'] = pd.read_hdf(os.path.join(path, 'Beam2xyz.h5'))
    Data['BlankingDistance'] = pd.read_hdf(os.path.join(path, 'EchoBlankingDistance.h5'))
    Data['EchoCellSize'] = pd.read_hdf(os.path.join(path, 'EchoCellSize.h5'))
    Data['EchoFrequency'] = pd.read_hdf(os.path.join(path, 'EchoFrequency.h5'))
    Data['NCells'] = pd.read_hdf(os.path.join(path, 'Burst_NCells.h5'))
    Data['CellSize'] = pd.read_hdf(os.path.join(path, 'Burst_CellSize.h5'))
    Data['SampleRate'] = pd.read_hdf(os.path.join(path, 'Burst_SampleRate.h5'))
    Data['VbAmplitude'] = pd.read_hdf(os.path.join(path, 'Burst_VertAmplitude.h5'))
    Data["Time"] = pd.DataFrame(dtnum_dttime_adcp(datenum_array.to_numpy()[0]))
    

    # Get individual beams
    number_vertical_cells = Data['NCells'][0][0]
    
    # Get individual beams
    Data["VelBeam1"] = (Data["VelBeam"].iloc[:, 0:number_vertical_cells])
    Data["VelBeam2"] = (Data["VelBeam"].iloc[:, number_vertical_cells:number_vertical_cells * 2])
    Data['VelBeam2'].reset_index(drop=True, inplace=True)  # KA: not sure if this is needed
    Data['VelBeam2'].columns = range(Data['VelBeam2'].columns.size)  # resets the column number
    Data["VelBeam3"] = (Data["VelBeam"].iloc[:, number_vertical_cells * 2:number_vertical_cells * 3])
    Data['VelBeam3'].reset_index(drop=True, inplace=True)
    Data['VelBeam3'].columns = range(Data['VelBeam3'].columns.size)
    Data["VelBeam4"] = (Data["VelBeam"].iloc[:, number_vertical_cells * 3:number_vertical_cells * 4])
    Data['VelBeam4'].reset_index(drop=True, inplace=True)
    Data['VelBeam4'].columns = range(Data['VelBeam4'].columns.size)


    # Get individual beams
    Data["CorBeam1"] = (Data["CorBeam"].iloc[:, 0:number_vertical_cells])
    Data["CorBeam2"] = (Data["CorBeam"].iloc[:, number_vertical_cells:number_vertical_cells * 2])
    Data['CorBeam2'].reset_index(drop=True, inplace=True)  # KA: not sure if this is needed
    Data['CorBeam2'].columns = range(Data['CorBeam2'].columns.size)  # resets the column number
    Data["CorBeam3"] = (Data["CorBeam"].iloc[:, number_vertical_cells * 2:number_vertical_cells * 3])
    Data['CorBeam3'].reset_index(drop=True, inplace=True)
    Data['CorBeam3'].columns = range(Data['CorBeam3'].columns.size)
    Data["CorBeam4"] = (Data["CorBeam"].iloc[:, number_vertical_cells * 3:number_vertical_cells * 4])
    Data['CorBeam4'].reset_index(drop=True, inplace=True)
    Data['CorBeam4'].columns = range(Data['CorBeam4'].columns.size)
    

    #Get individual beams
    Data["AmpBeam1"] = (Data["AmpBeam"].iloc[:, 0:number_vertical_cells])
    Data["AmpBeam2"] = (Data["AmpBeam"].iloc[:, number_vertical_cells:number_vertical_cells * 2])
    Data['AmpBeam2'].reset_index(drop=True, inplace=True)  # KA: not sure if this is needed
    Data['AmpBeam2'].columns = range(Data['AmpBeam2'].columns.size)  # resets the column number
    Data["AmpBeam3"] = (Data["AmpBeam"].iloc[:, number_vertical_cells * 2:number_vertical_cells * 3])
    Data['AmpBeam3'].reset_index(drop=True, inplace=True)
    Data['AmpBeam3'].columns = range(Data['AmpBeam3'].columns.size)
    Data["AmpBeam4"] = (Data["AmpBeam"].iloc[:, number_vertical_cells * 3:number_vertical_cells * 4])
    Data['AmpBeam4'].reset_index(drop=True, inplace=True)
    Data['AmpBeam4'].columns = range(Data['AmpBeam4'].columns.size)
    
    

    # Create cell depth vector
    vector = np.arange(1, Data["NCells"][0][0] + 1)

    # Calculate the depth of each cell
    Data["CellDepth"] = (
            Data["BlankingDistance"][0].iloc[0]
            + vector * Data["CellSize"][0].iloc[0]
    )

    #Create echo cell depth vector
    Data['EchoNCells'] = Data['Echo1'].shape[1]  # Number of echo cells
    print(f"Number of echo cells: {Data['EchoNCells']}")  # debugging line
    vector = np.arange(1, Data["EchoNCells"].iloc[0] + 1)

    # Calculate the depth of each cell
    Data["CellDepth_echo"] = (
            Data["BlankingDistance"][0].iloc[0]
            + vector * Data["EchoCellSize"][0].iloc[0]
    )

    print(f"Sample VelBeam1 values: {Data['VelBeam1'].shape}")  # debugging line
    return Data

def remove_low_correlations(Data):
    """
    The first step in data quality control is to remove any low-correlation data points across all depths.
    This is accomplished using the equation Threshold = .3+.4*sqrt(sf/25) from Elga (2001), which incorporates the 
    instrument sample rate. For s 4 Hz measurements, the threshold is approximately 0.46, so data points with
    correlation values lower than 0.46 will be flagged and converted to NaNs.

    Next, data collected beyond the water surface is removed. We set a water depth limit based on the highest tide,
    but at lower tides, data collected above the surface must be discarded. This is done using pressure readings
    directly from the instrument. Eventually, we will try using the 5th beam for this purpose.

    :param:
    Data: dictionary
        data data as a dictionary

    :return:
    Data: dictionary
        data data as a dictionary

    """

    # Get the dimensions of the matrices
    row, col = Data["VelBeam1"].to_numpy().shape

    Sr = Data["SampleRate"][0].iloc[0]  # Sample rate in Hz

    CorrThresh = (
            0.3 + 0.4 * (Sr / 25) ** 0.5
    )  # Threshold for correlation values as found in Elgar
    
    row, col = Data['Burst_VertAmplitude'].shape
    row1, col2 = Data['Echo1'].shape


    isbad = np.zeros((row, col))  # Initialize mask for above surface measurements
    echobad = np.zeros((row1, col2))
    print('test')
    
    # Apply mask for surface measurements
    for i in range(len(isbad)):
        Depth_Thresh = (
                Data["Pressure"].iloc[i][0] * np.cos(25 * np.pi / 180)
                - Data["CellSize"][0]
        )
        isbad[i, :] = Data["CellDepth"] >= Depth_Thresh
    isbad = isbad.astype(bool)
    Data["DepthThresh"] = isbad

    for jj in range(1, 5):
        isbad2 = (
                Data[f"CorBeam{jj}"] * 0.01 <= CorrThresh
        )  # create mask for bad correlations
        isbad2 = isbad2.astype(bool)
        Data[f"VelBeam{jj}"] = Data[f"VelBeam{jj}"].mask(isbad, np.nan)
        Data[f"VelBeam{jj}"] = Data[f"VelBeam{jj}"].mask(isbad2, np.nan)
        Data[f"AmpBeam{jj}"] = Data[f"AmpBeam{jj}"].mask(isbad, np.nan)
        Data[f"AmpBeam{jj}"] = Data[f"AmpBeam{jj}"].mask(isbad2, np.nan)
        Data[f"VelBeamCorr{jj}"] = isbad2

    # Apply mask for surface measurements for the echo sounders 
    for i in range(len(echobad)):
        Depth_Thresh1 = (
            Data["Burst_Pressure"].iloc[i][0] * np.cos(25 * np.pi / 180)
            - Data["EchoCellSize"][0]
        )
        echobad[i, :] = Data["CellDepth_echo"] >= Depth_Thresh1
    echobad = echobad.astype(bool)
    Data["EchoDepthThresh"] = echobad

        # Mask the data 
    Data["Echo1"] = Data["Echo1"].mask(echobad, np.nan)
    Data["Echo2"] = Data["Echo2"].mask(echobad, np.nan)

    #mask the Data
    Data[f"Burst_VertAmplitude"] = Data[f"Burst_VertAmplitude"].mask(isbad, np.nan)

    return Data



def transform_beam_ENUD(Data):
    """
    Data is then converted from beam coordinates to ENUD (East, North, Up, Difference) using the transformation
    matrix provided in the exported .mat file. The matrix incorporates beam angles and directions, and is combined
    with heading, roll, and pitch data for the transformation.

    https://support.nortekgroup.com/hc/en-us/articles/360029820971-How-is-a-coordinate-transformation-done

    :param:
    Data: dictionary
        data data as a dictionary

    :return:
    Data: dictionary
        data data as a dictionary
    """
    
    # Load the transformation matrix
    T = pd.DataFrame(Data["Beam2xyz"]).to_numpy()
    
    # Transform attitude data to radians
    hh = np.pi * (Data["Heading"].to_numpy() - 90) / 180
    pp = np.pi * Data["Pitch"].to_numpy() / 180
    rr = np.pi * Data["Roll"].to_numpy() / 180

    # Create the tiled transformation matrix, this is for applying the transformation later to each data point
    row, col = Data["VelBeam1"].to_numpy().shape  # Get the dimensions of the matrices
    
    Tmat = np.tile(T, (row, 1, 1))
    
    # Initialize heading and tilt matrices
    Hmat = np.zeros((3, 3, row))
    Pmat = np.zeros((3, 3, row))
    
    
    # Using vector mat populate the heading matrix and pitch/roll matrix with the appropriate values
    # The 3x3xrow matrix is the spatial dimensios at each measurement
    for i in range(row):
        Hmat[:, :, i] = [
            [np.cos(hh[i][0]), np.sin(hh[i][0]), 0],
            [-np.sin(hh[i][0]), np.cos(hh[i][0]), 0],
            [0, 0, 1],
        ]

        Pmat[:, :, i] = [
            [
                np.cos(pp[i][0]),
                -np.sin(pp[i][0]) * np.sin(rr[i][0]),
                -np.cos(rr[i][0]) * np.sin(pp[i][0]),
            ],
            [0, np.cos(rr[i][0]), -np.sin(rr[i][0])],
            [
                np.sin(pp[i][0]),
                np.sin(rr[i][0]) * np.cos(pp[i][0]),
                np.cos(pp[i][0]) * np.cos(rr[i][0]),
            ],
        ]
    print('testest')    
    # Combine the Hmat and Pmat vectors into one rotation matrix, this conversion matrix is organized with beams in the
    # columns and the rotation values on the rows (for each data point). The original Hmat and Pmat matrices are only
    # made with the one Z value in mind so we duplicate the 4 row of the transform matirx to create the fourth, same
    # process for fourth column.
    #                     Beam1   Beam2   Beam3   Beam4
    #                X   [                               ]
    #                Y   [                               ]             (at nth individual sample)
    #               Z1   [                          0    ]
    #               Z2   [                  0            ]
    
    R1Mat = np.zeros((4, 4, row))  # initialize rotation matrix
    print(Hmat.shape, Pmat.shape)  # debugging line
    for i in range(row):
        R1Mat[0:3, 0:3, i] = Hmat[:, :, i] @ Pmat[:, :, i]  # Matrix multiplication
        R1Mat[3, 0:4, i] = R1Mat[2, 0:4, i]  # Create fourth row
        R1Mat[0:4, 3, i] = R1Mat[0:4, 2, i]  # Create fourth column

    # We zero out these value since Beams 3 and 4 can't measure both Z's
    R1Mat[2, 3, :] = 0
    R1Mat[3, 2, :] = 0
    print(R1Mat.shape)  # debugging line
    Rmat = np.zeros((4, 4, row))

    Tmat = np.swapaxes(Tmat, 0, -1)
    Tmat = np.swapaxes(Tmat, 0, 1)

    print(Tmat.shape)  # debugging line
    for i in range(row):
        Rmat[:, :, i] = R1Mat[:, :, i] @ Tmat[:, :, i]

    Velocities = np.squeeze(
        np.array(
            [
                [Data["VelBeam1"]],
                [Data["VelBeam2"]],
                [Data["VelBeam3"]],
                [Data["VelBeam4"]],
            ]
        )
    )
    print(Rmat.shape, Velocities.shape)  # debugging line
    print(Tmat.shape)  # debugging line
    # Convert to ENU
    ENU = np.einsum("ijk,jkl->ikl", Rmat, Velocities)
    ENU = np.transpose(ENU, (1, 2, 0))
    Data["ENU"] = ENU
    del ENU

    Data["ENU"][:, :, 3] = abs(Data["ENU"][:, :, 2] - Data["ENU"][:, :, 3])

    Data['EastVel'] = pd.DataFrame(Data['ENU'][:, :, 0])
    Data['NorthVel'] = pd.DataFrame(Data['ENU'][:, :, 1])
    Data['VertVel'] = pd.DataFrame(Data['ENU'][:, :, 2])
    Data['ErrVel'] = pd.DataFrame(Data['ENU'][:, :, 3])
    # print(f"Sample EastVel values: {Data['EastVel'].head()}") debugging line
    
    # Add matrices with NaN values together treating nan values as 0, this is for calculating the absolute velocity
    nan_mask = np.full((row, col), False)

    for i in range(col):
        nan_mask[:, i] = (
                np.isfinite(Data["ENU"][:, i, 0])
                & np.isfinite(Data["ENU"][:, i, 1])
                & np.isfinite(Data["ENU"][:, i, 2])
        )

    # Replace NaNs with zeroes for the calculation
    NorthVel_no_nan = np.nan_to_num(Data["ENU"][:, :, 0], nan=0.0)
    EastVel_no_nan = np.nan_to_num(Data["ENU"][:, :, 1], nan=0.0)
    VertVel_no_nan = np.nan_to_num(Data["ENU"][:, :, 2], nan=0.0)

    # Sum the squared velocities
    Data["AbsVel"] = pd.DataFrame(
        np.sqrt(NorthVel_no_nan ** 2 + EastVel_no_nan ** 2 + VertVel_no_nan ** 2)
    )
    print(Data["Time"])
    # Reapply the mask to set positions with any original NaNs back to NaN
    Data["AbsVel"][~nan_mask] = np.nan
    Data['CellDepth'] = pd.DataFrame(Data['CellDepth'])

    # print(f"AbsVel shape: {Data['AbsVel'].shape}")
    # print(f"Sample AbsVel values: {Data['AbsVel'].head()}")
    return Data

def save_data(Data, save_dir):
    """
    Saves h5 files of quality controlled data from Sig1000.

    :param:
    Data: dictionary
        data data as a dictionary
    save_dir: string
        path to directory where processed, quality controlled data should be saved

    :return:
    none

    """
    print("Saving data...")
    # # Open the HDF5 file in write mode
    # file_path = os.path.join(save_dir, 'DepthThresh.h5')
    # with h5py.File(file_path, 'w') as f:
    #     # Save the NumPy array under the key 'df'
    #     f.create_dataset('df', data=Data['DepthThresh'])

    # Save the data fields
    Data['AbsVel'].to_hdf(
        os.path.join(save_dir, 'AbsVel.h5'), key="df", mode="w"
    )
    Data['Time'].to_hdf(
        os.path.join(save_dir, 'Time.h5'), key="df", mode="w"
    )
    Data['EastVel'].to_hdf(
        os.path.join(save_dir, 'EastVel.h5'), key="df", mode="w"
    )
    Data['NorthVel'].to_hdf(
        os.path.join(save_dir, 'NorthVel.h5'), key="df", mode="w"
    )
    Data['VertVel'].to_hdf(
        os.path.join(save_dir, 'VertVel.h5'), key="df", mode="w"
    )
    Data['ErrVel'].to_hdf(
        os.path.join(save_dir, 'ErrVel.h5'), key="df", mode="w"
    )
    Data['Heading'].to_hdf(
        os.path.join(save_dir, 'Heading.h5'), key="df", mode="w"
    )
    Data['Roll'].to_hdf(
        os.path.join(save_dir, 'Roll.h5'), key="df", mode="w"
    )
    Data['Pitch'].to_hdf(
        os.path.join(save_dir, 'Pitch.h5'), key="df", mode="w"
    )
    Data['Pressure'].to_hdf(
        os.path.join(save_dir, 'Pressure.h5'), key="df", mode="w"
    )
    Data['VelBeamCorr1'].to_hdf(
        os.path.join(save_dir, 'VelBeamCorr1.h5'), key="df", mode="w"
    )
    Data['VelBeamCorr2'].to_hdf(
        os.path.join(save_dir, 'VelBeamCorr2.h5'), key="df", mode="w"
    )
    Data['VelBeamCorr3'].to_hdf(
        os.path.join(save_dir, 'VelBeamCorr3.h5'), key="df", mode="w"
    )
    Data['VelBeamCorr4'].to_hdf(
        os.path.join(save_dir, 'VelBeamCorr4.h5'), key="df", mode="w"
    )
    Data['CellDepth'].to_hdf(os.path.join(save_dir, 'CellDepth.h5'), key="df", mode="w")
    Data['Echo1'].to_hdf(os.path.join(save_dir, 'Echo1.h5'), key="df", mode="w")
    Data['Echo2'].to_hdf(os.path.join(save_dir, 'Echo2.h5'), key="df", mode="w")
    Data['CellDepth_echo'].to_hdf(os.path.join(save_dir, 'CellDepth_echo.h5'), key="df", mode="w")
    Data['AmpBeam1'].to_hdf(os.path.join(save_dir, 'AmpBeam1.h5'), key="df", mode="w")
    Data['AmpBeam2'].to_hdf(os.path.join(save_dir, 'AmpBeam2.h5'), key="df", mode="w")
    Data['AmpBeam3'].to_hdf(os.path.join(save_dir, 'AmpBeam3.h5'), key="df", mode="w")
    Data['AmpBeam4'].to_hdf(os.path.join(save_dir, 'AmpBeam4.h5'), key="df", mode="w")
    Data['VbAmplitude'].to_hdf(os.path.join(save_dir, 'VbAmplitude.h5'), key="df", mode="w")

    return


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
    files.sort(key=lambda x: int(re.search(r"FPS4_(\d+)", x).group(1)) if re.search(r"FPS4_(\d+)", x) else float('inf'))

    file_id = group_id - 1

    for file_name in files[file_id:]:
        file_id += 1
        path = os.path.join(directory_path_mat, file_name)
        print(path)
        if file_id < 10:
            save_path = os.path.join(save_dir_data, f"Group0{file_id}")
        else:
            save_path = os.path.join(save_dir_data, f"Group{file_id}")
        #read_Sig1k(path, config_path ,save_path) #Why is here, this makes it read twice?
        try:
            read_Sig1k(path, config_path,save_path)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")


###############################################################################
# quality control
###############################################################################
if run_quality_control:
    print("Running Quality Control")

    files = sorted(os.listdir(save_dir_data))

    if '.DS_Store' in files:  # remove hidden files on macs
        files.remove('.DS_Store')
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
if run_bulk_statistics:
    waves = bulk_stats_analysis(save_dir_qc, save_dir_bulk_stats, group_ids_exclude)

print(waves['DepthAveragedCurrentVelocity'])
print(waves['SignificantWaveHeight'])
print(waves['WaveDirection'])   
