from math import floor
import datetime as dt
import numpy as np
import pandas as pd
import os
import time

start_time = time.time()

def dtnum_dttime_adcp(
    time_array,
):  # Create a function to convert from matlabs datenum format to python datetime
    dates = []  # Initialize dates#Convert the awkward structure intoa matlab array
    DT = time_array.to_numpy()  # Dataframes behave strangely for some reason
    for ordinal in DT:
        integer = floor(
            ordinal[0]
        )  # Round down to the nearest whole number on the datetime
        frac = ordinal - integer  # Get just the decimal points
        date = dt.datetime.fromordinal(
            integer
        )  # This function only takes integer values so we must add the decimal on after the conversion
        time = dt.timedelta(
            days=frac[0]
        )  # Convert the decimals into hours, min, seconds, microseconds, ect.
        mat_correction = dt.timedelta(
            days=366
        )  # Matlab datenum starts from Jan 1 0000AD, this year is 366 days long,
        # while ordinal time which is what we are converting from starts Jan 1 0001AD
        full = (
            date + time - mat_correction
        )  # Recombine the fractional precision and correct fro the year difference between ordinal and dateum
        dates.append(full)  # Append the correct datetime back into the dates array
    return dates
    """This entire loop is very inefficient, will take time for large data sets
                     but this is the best I have figured out at the time of creating this"""


def process(path, save_dir):
    #Initialize the Data dictionary as well as it's keys.
    Data = {}
    
    Data['CorBeam'] = pd.read_hdf(os.path.join(path,'Burst_CorBeam.h5'))
    Data['Heading'] = pd.read_hdf(os.path.join(path,'Burst_Heading.h5'))
    Data['Pressure'] = pd.read_hdf(os.path.join(path,'Burst_Pressure.h5'))
    Data['Roll'] = pd.read_hdf(os.path.join(path,'Burst_Roll.h5'))
    Data['Time'] = pd.read_hdf(os.path.join(path,'Burst_Time.h5'))
    Data['VelBeam'] = pd.read_hdf(os.path.join(path,'Burst_VelBeam.h5'))
    Data['Pitch'] = pd.read_hdf(os.path.join(path,'Burst_Pitch.h5'))

    Data['Beam2xyz'] = pd.read_hdf(os.path.join(path,'Burst_Beam2xyz.h5'))
    Data['BlankingDistance'] = pd.read_hdf(os.path.join(path,'Burst_BlankingDistance.h5'))
    Data['CellSize'] = pd.read_hdf(os.path.join(path,'Burst_CellSize.h5'))
    Data['NCells'] = pd.read_hdf(os.path.join(path,'Burst_NCells.h5'))
    Data['SampleRate'] = pd.read_hdf(os.path.join(path,'Burst_SamplingRate.h5'))
    Data["Time"] = pd.DataFrame(dtnum_dttime_adcp(Data["Time"]))

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
    
    # Get the dimensions of the matrices
    row, col = Data["VelBeam1"].to_numpy().shape

    # Create cell depth vector
    vector = np.arange(1, Data["NCells"][0][0] + 1)
   
    Data["CellDepth"] = (
        Data["BlankingDistance"][0].iloc[0]
        + vector * Data["CellSize"][0].iloc[0]
    )
    
    # QC the data based on correlation values based on the info in Elgar 2001 and
    # remove data thats collected over the surface of the water and remove the influence of the side lobes.
    # We will do this according to the processes in the ADCP Comprenhensive Manual from Nortek.
    Sr = Data["SampleRate"][0].iloc[0]  # Sample rate in Hz
    CorrThresh = (
        0.3 + 0.4 * (Sr / 25) ** 0.5
    )  # Threshold for correlation values as found in Elgar
    isbad = np.zeros((row, col))  # Initialize mask for above surface measurements

    for i in range(len(isbad)):
        Depth_Thresh = (
            Data["Pressure"].iloc[i][0] * np.cos(25 * np.pi / 180)
            - Data["CellSize"][0].iloc[0]
        )
        isbad[i, :] = Data["CellDepth"] >= Depth_Thresh
    isbad = isbad.astype(np.bool_)

    for jj in range(1, 5):
        isbad2 = (
            Data[f"CorBeam{jj}"] * 0.01 <= CorrThresh
        )  # create mask for bad correlations
        Data[f"VelBeam{jj}"][isbad] = np.nan
        Data[f"VelBeam{jj}"][isbad2] = np.nan

    # Convert the data from beam coords to ENU (This is all done according to the steps found here
    # https://support.nortekgroup.com/hc/en-us/articles/360029820971-How-is-a-coordinate-transformation-done

    # Load the transformation matrix
    T = pd.DataFrame(Data["Beam2xyz"]).to_numpy()

    # Transform attitude data to radians
    hh = np.pi * (Data["Heading"].to_numpy() - 90) / 180
    pp = np.pi * Data["Pitch"].to_numpy() / 180
    rr = np.pi * Data["Roll"].to_numpy() / 180

    # Create the tiled transformation matrix, this is for applyinh the transformation later to each data point
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

    # Combine the Hmat and Pmat vectors into one rotation matrix, this conversion matrix is organized with beams in the columns
    # and the rotation values on the rows (for each data point). The original Hmat and Pmat matrices are only made with the one Z
    # value in mind so we duplicate the 4 row of the transform matirx to create the fourth, same process for fourth column.
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

    ### We zero out these value since Beams 3 and 4 can't measure both Z's
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

    # Add matrices with NaN values together treating nan values as 0, this for calculating the absolute velocity
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
        np.sqrt(NorthVel_no_nan**2 + EastVel_no_nan**2 + VertVel_no_nan**2)
    )

    # Reapply the mask to set positions with any original NaNs back to NaN
    Data["AbsVel"][~nan_mask] = np.nan

    Data['CellDepth'] = pd.DataFrame(Data['CellDepth'])

    #Save the data gields
    Data['AbsVel'].to_hdf(
        os.path.join(save_dir, 'AbsVel'), key="df", mode="w"
    )
    Data['Time'].to_hdf(
        os.path.join(save_dir, 'Time'), key="df", mode="w"
    )
    Data['EastVel'].to_hdf(
        os.path.join(save_dir, 'EastVel'), key="df", mode="w"
    )
    Data['NorthVel'].to_hdf(
        os.path.join(save_dir, 'NorthVel'), key="df", mode="w"
    )
    Data['VertVel'].to_hdf(
        os.path.join(save_dir, 'VertVel'), key="df", mode="w"
    )
    Data['ErrVel'].to_hdf(
        os.path.join(save_dir, 'ErrVel'), key="df", mode="w"
    )
    Data['CellDepth'].to_hdf( os.path.join(save_dir, 'CellDepth'), key="df", mode="w")
    
directory_path = r"Z:\BHBoemData\Raw\S0_103080_hdf\Group2"
save_dir = r"Z:\BHBoemData\Processed\S0_103080"

process(directory_path, save_dir)

endtime = time.time()

print("Time taken was", endtime-start_time, "seconds")