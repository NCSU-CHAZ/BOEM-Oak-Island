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
from datetime import datetime, timedelta


def dtnum_dttime_adcp(datenum_array):
    dates = []
    for datenum in datenum_array:
        python_datetime = datetime.fromordinal(int(datenum)) + timedelta(days=datenum % 1) - timedelta(days=366)
        dates.append(python_datetime)
    return dates


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
    Data['Pitch'] = pd.read_hdf(os.path.join(path, 'Burst_Pitch.h5'))

    Data['Beam2xyz'] = pd.read_hdf(os.path.join(path, 'Burst_Beam2xyz.h5'))
    Data['BlankingDistance'] = pd.read_hdf(os.path.join(path, 'Burst_BlankingDistance.h5'))
    Data['CellSize'] = pd.read_hdf(os.path.join(path, 'Burst_CellSize.h5'))
    Data['NCells'] = pd.read_hdf(os.path.join(path, 'Burst_NCells.h5'))
    Data['SampleRate'] = pd.read_hdf(os.path.join(path, 'Burst_SamplingRate.h5'))
    Data["Time"] = pd.DataFrame(dtnum_dttime_adcp(datenum_array[0].values))

    # Get individual beams
    Data["VelBeam1"] = (Data["VelBeam"].iloc[:, 0::4])
    Data['VelBeam1'].reset_index(drop=True, inplace=True)
    Data["VelBeam2"] = (Data["VelBeam"].iloc[:, 1::4])
    Data['VelBeam2'].reset_index(drop=True, inplace=True)
    Data["VelBeam3"] = (Data["VelBeam"].iloc[:, 2::4])
    Data['VelBeam3'].reset_index(drop=True, inplace=True)
    Data["VelBeam4"] = (Data["VelBeam"].iloc[:, 3::4])
    Data['VelBeam4'].reset_index(drop=True, inplace=True)

    # Get individual beams
    Data["CorBeam1"] = (Data["CorBeam"].iloc[:, 0::4])
    Data['CorBeam1'].reset_index(drop=True, inplace=True)
    Data["CorBeam2"] = (Data["CorBeam"].iloc[:, 1::4])
    Data['CorBeam2'].reset_index(drop=True, inplace=True)
    Data["CorBeam3"] = (Data["CorBeam"].iloc[:, 2::4])
    Data['CorBeam3'].reset_index(drop=True, inplace=True)
    Data["CorBeam4"] = (Data["CorBeam"].iloc[:, 3::4])
    Data['CorBeam4'].reset_index(drop=True, inplace=True)

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

    # isbad2 = np.zeros((row, col))  # initialize mask

    for jj in range(1, 5):
        isbad2 = (
                Data[f"CorBeam{jj}"] * 0.01 <= CorrThresh
        )  # create mask for bad correlations
        isbad2 = isbad2.astype(bool)
        # Initialize the VelBeamCorr{jj} columns with zeros
        Data[f"VelBeamCorr{jj}"] = pd.DataFrame(np.zeros((row, col)), index=Data[f"CorBeam{jj}"].index)
        Data[f"VelBeam{jj}"] = Data[f"VelBeam{jj}"].mask(isbad, np.nan)
        Data[f"VelBeam{jj}"] = Data[f"VelBeam{jj}"].mask(isbad2, np.nan)
        Data[f"VelBeamCorr{jj}"] = Data[f"VelBeamCorr{jj}"].mask(isbad, 1)  # Set to 1 for bad correlations
        Data[f"VelBeamCorr{jj}"] = Data[f"VelBeamCorr{jj}"].mask(isbad2, 1)  # Set to 1 for bad correlations
        


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

    # Create the tiled transformation matrix, this is for applyinh the transformation later to each data point
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
    print(f"Sample EastVel values: {Data['EastVel'].head()}")

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

    return
