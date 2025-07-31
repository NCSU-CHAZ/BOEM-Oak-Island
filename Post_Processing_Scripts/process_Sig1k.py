"""Perform quality control on raw data from the Nortek Signature 1000 (upward-looking ADCP)

The following functions post-process raw data from a Nortek ADCP for easy processing in python and conduct basic
data quality control.

References
----------

.. [1]
.. [2]


Notes
---------

"""

import numpy as np
import pandas as pd
import os
import h5py
from datetime import datetime, timedelta
from scipy.io import loadmat


def dtnum_dttime_adcp(datenum_array):
    dates = []
    for datenum in datenum_array:
        python_datetime = datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)
        dates.append(python_datetime)
    return dates


def read_Sig1k(filepath, save_dir):  # Create read function
    # loadmat organizes the 4 different data structures of the .mat file (Units, Config, Data, Description) as a
    # dictionary with four nested numpy arrays with dtypes as data field titles
    Data = loadmat(
        filepath
    )
    ADCPData = {}  # Initialize the dictionary we'll use
    Config = Data["Config"][0, 0]

    # Save BEAM coordinate velocity matrix
    VelArray = Data["Data"][0, 0]["Burst_Velocity_Beam"]
    reshaped = VelArray.reshape(VelArray.shape[0], -1)
    del VelArray
    ADCPData["Burst_VelBeam"] = pd.DataFrame(reshaped)

    # Save Vertical Amplitude coordinate velocity matrix
    VelArray = Data["Data"][0, 0]["IBurst_Amplitude_Beam"]
    reshaped = VelArray.reshape(VelArray.shape[0],-1)
    print(reshaped.shape)
    del VelArray
    ADCPData['Burst_VertAmplitude'] = pd.DataFrame(reshaped)
    print('saved VertAmp')

    # Save the correlation data matrix
    CorArray = Data["Data"][0, 0]["Burst_Correlation_Beam"]
    reshaped = CorArray.reshape(CorArray.shape[0], -1)
    ADCPData["Burst_CorBeam"] = pd.DataFrame(reshaped)
    del CorArray, reshaped

    # Save transformed velocity matrix (used for testing)
    VelArray = Data["Data"][0, 0]["Burst_Velocity_ENU"]
    reshaped = VelArray.reshape(VelArray.shape[0], -1)
    del VelArray
    ADCPData["Burst_ENU"] = pd.DataFrame(reshaped)

    # Save other fields
    ADCPData["Burst_Time"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Time"])
    ADCPData["Burst_NCells"] = pd.DataFrame(Data["Data"][0, 0]["Burst_NCells"])
    ADCPData["Burst_Pressure"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Pressure"])
    ADCPData["Burst_Heading"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Heading"])
    ADCPData["Burst_Pitch"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Pitch"])
    ADCPData["Burst_Roll"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Roll"])
    ADCPData["Burst_Pitch"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Pitch"])
    # Fifth Beam
    ADCPData['Burst_AltimeterDistanceAST']=pd.DataFrame(Data["Data"][0,0]["Burst_AltimeterDistanceAST"]) # this should save for each burst
    print("AST is",ADCPData["Burst_AltimeterDistanceAST"]) # debugging
    print('saved AST dist')

    BlankDist = pd.DataFrame(Config["Burst_BlankingDistance"])
    CellSize = pd.DataFrame(Config["Burst_CellSize"])
    SampleRate = pd.DataFrame(Config["Burst_SamplingRate"])
    Beam2xyz = pd.DataFrame(Config["Burst_Beam2xyz"])

    # Make directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save files there
    BlankDist.to_hdf(
        os.path.join(save_dir, "Burst_BlankingDistance.h5"), key="df", mode="w"
    )
    CellSize.to_hdf(os.path.join(save_dir, "Burst_CellSize.h5"), key="df", mode="w")
    SampleRate.to_hdf(
        os.path.join(save_dir, "Burst_SamplingRate.h5"), key="df", mode="w"
    )
    Beam2xyz.to_hdf(os.path.join(save_dir, "Burst_Beam2xyz.h5"), key="df", mode="w")

    for field_name, df in ADCPData.items():
        save_path = os.path.join(save_dir, f"{field_name}.h5")
        df.to_hdf(save_path, key="df", mode="w")
        print(f"Saved {field_name} to {save_path}")

    print("Saving Done")


def read_raw_h5(path):
    """
    Read h5 files of raw data from Sig1000. Raw data converted from mat to h5 in 'read_Sig1k'

    :param:
    path: string
        path to directory to raw h5 files
    save_dir: string
        path to directory where processed, quality controlled data should be saved

    :return:
    Data: dictionary
        raw data as a dictionary
    """

    # initialize the Data dictionary as well as it's keys
    Data = {}

    Data['CorBeam'] = pd.read_hdf(os.path.join(path, 'Burst_CorBeam.h5'))
    Data['Heading'] = pd.read_hdf(os.path.join(path, 'Burst_Heading.h5'))
    Data['Pressure'] = pd.read_hdf(os.path.join(path, 'Burst_Pressure.h5'))
    Data['Roll'] = pd.read_hdf(os.path.join(path, 'Burst_Roll.h5'))
    datenum_array = pd.read_hdf(os.path.join(path, 'Burst_Time.h5'))
    Data['VelBeam'] = pd.read_hdf(os.path.join(path, 'Burst_VelBeam.h5'))
    Data['Vel_ENU_mat'] = pd.read_hdf(os.path.join(path, 'Burst_ENU.h5'))
    Data['Pitch'] = pd.read_hdf(os.path.join(path, 'Burst_Pitch.h5'))

    Data['Beam2xyz'] = pd.read_hdf(os.path.join(path, 'Burst_Beam2xyz.h5'))
    Data['BlankingDistance'] = pd.read_hdf(os.path.join(path, 'Burst_BlankingDistance.h5'))
    Data['CellSize'] = pd.read_hdf(os.path.join(path, 'Burst_CellSize.h5'))
    Data['NCells'] = pd.read_hdf(os.path.join(path, 'Burst_NCells.h5'))
    Data['SampleRate'] = pd.read_hdf(os.path.join(path, 'Burst_SamplingRate.h5'))
    Data["Time"] = pd.DataFrame(dtnum_dttime_adcp(datenum_array[0].values))

    # Get Fifth Beam
    Data['Altimeter_DistAST'] = pd.read_hdf(os.path.join(path, 'Burst_AltimeterDistanceAST.h5'))
    Data['VertAmp'] = pd.read_hdf(os.path.join(path, 'Burst_VertAmplitude.h5'))

    # Get individual beams
    number_vertical_cells = Data['NCells'][0][0]
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
    # Data["VelBeam1"] = (Data["VelBeam"].iloc[:, 0::4])
    # Data['VelBeam1'].reset_index(drop=True, inplace=True)
    # Data["VelBeam2"] = (Data["VelBeam"].iloc[:, 1::4])
    # Data['VelBeam2'].reset_index(drop=True, inplace=True)
    # Data["VelBeam3"] = (Data["VelBeam"].iloc[:, 2::4])
    # Data['VelBeam3'].reset_index(drop=True, inplace=True)
    # Data["VelBeam4"] = (Data["VelBeam"].iloc[:, 3::4])
    # Data['VelBeam4'].reset_index(drop=True, inplace=True)

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
    # Data["CorBeam1"] = (Data["CorBeam"].iloc[:, 0::4])
    # Data['CorBeam1'].reset_index(drop=True, inplace=True)
    # Data["CorBeam2"] = (Data["CorBeam"].iloc[:, 1::4])
    # Data['CorBeam2'].reset_index(drop=True, inplace=True)
    # Data["CorBeam3"] = (Data["CorBeam"].iloc[:, 2::4])
    # Data['CorBeam3'].reset_index(drop=True, inplace=True)
    # Data["CorBeam4"] = (Data["CorBeam"].iloc[:, 3::4])
    # Data['CorBeam4'].reset_index(drop=True, inplace=True)

    # Get individual beams ENU for testing
    Data["VelE_mat"] = (Data["Vel_ENU_mat"].iloc[:, 0:number_vertical_cells])
    Data["VelN_mat"] = (Data["Vel_ENU_mat"].iloc[:, number_vertical_cells:number_vertical_cells * 2])
    Data['VelN_mat'].reset_index(drop=True, inplace=True)  # KA: not sure if this is needed
    Data['VelN_mat'].columns = range(Data['VelN_mat'].columns.size)  # resets the column number
    Data["VelU_mat"] = (Data["Vel_ENU_mat"].iloc[:, number_vertical_cells * 2:number_vertical_cells * 3])
    Data['VelU_mat'].reset_index(drop=True, inplace=True)
    Data['VelU_mat'].columns = range(Data['VelU_mat'].columns.size)
    Data["VelDiff_mat"] = (Data["Vel_ENU_mat"].iloc[:, number_vertical_cells * 3:number_vertical_cells * 4])
    Data['VelDiff_mat'].reset_index(drop=True, inplace=True)
    Data['VelDiff_mat'].columns = range(Data['VelDiff_mat'].columns.size)
    # Data["VelE_mat"] = (Data["Vel_ENU_mat"].iloc[:, 0::4])  # this is wrong, they should be sequential [0-29], [30-59], [60-etc]..
    # Data["VelN_mat"] = (Data["Vel_ENU_mat"].iloc[:, 1::4])
    # Data["VelU_mat"] = (Data["Vel_ENU_mat"].iloc[:, 2::4])
    # Data["VelDiff_mat"] = (Data["Vel_ENU_mat"].iloc[:, 3::4])

    # Create cell depth vector
    vector = np.arange(1, Data["NCells"][0][0] + 1)

    Data["CellDepth"] = (
            Data["BlankingDistance"][0].iloc[0]
            + vector * Data["CellSize"][0].iloc[0]
    )
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
        raw data as a dictionary

    :return:
    Data: dictionary
        raw data as a dictionary

    """

    # Get the dimensions of the matrices
    row, col = Data["VelBeam1"].to_numpy().shape

    Sr = Data["SampleRate"][0].iloc[0]  # Sample rate in Hz

    CorrThresh = (
            0.3 + 0.4 * (Sr / 25) ** 0.5
    )  # Threshold for correlation values as found in Elgar

    isbad = np.zeros((row, col))  # Initialize mask for above surface measurements

    # Apply mask for surface measurements
    for i in range(len(isbad)):
        Depth_Thresh = (
                Data["Pressure"].iloc[i][0] * np.cos(25 * np.pi / 180)
                - Data["CellSize"][0].iloc[0]
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
        Data[f"VelBeamCorr{jj}"] = isbad2

    return Data


def transform_beam_ENUD(Data):
    """
    Data is then converted from beam coordinates to ENUD (East, North, Up, Difference) using the transformation
    matrix provided in the exported .mat file. The matrix incorporates beam angles and directions, and is combined
    with heading, roll, and pitch data for the transformation.

    https://support.nortekgroup.com/hc/en-us/articles/360029820971-How-is-a-coordinate-transformation-done

    :param:
    Data: dictionary
        raw data as a dictionary

    :return:
    Data: dictionary
        raw data as a dictionary
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

    for i in range(row):
        R1Mat[0:3, 0:3, i] = Hmat[:, :, i] @ Pmat[:, :, i]  # Matrix multiplication
        R1Mat[3, 0:4, i] = R1Mat[2, 0:4, i]  # Create fourth row
        R1Mat[0:4, 3, i] = R1Mat[0:4, 2, i]  # Create fourth column

    # We zero out these value since Beams 3 and 4 can't measure both Z's
    R1Mat[2, 3, :] = 0
    R1Mat[3, 2, :] = 0

    Rmat = np.zeros((4, 4, row))

    Tmat = np.swapaxes(Tmat, 0, -1)
    Tmat = np.swapaxes(Tmat, 0, 1)

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
        raw data as a dictionary
    save_dir: string
        path to directory where processed, quality controlled data should be saved

    :return:
    none

    """
    # Open the HDF5 file in write mode
    file_path = os.path.join(save_dir, 'DepthThresh.h5')
    with h5py.File(file_path, 'w') as f:
        # Save the NumPy array under the key 'df'
        f.create_dataset('df', data=Data['DepthThresh'])

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
    Data['Altimeter_DistAST'].to_hdf(
        os.path.join(save_dir, 'Burst_AltimeterDistanceAST.h5'), key="df", mode="w"
    )
    Data['CellDepth'].to_hdf(os.path.join(save_dir, 'CellDepth.h5'), key="df", mode="w")

    return
