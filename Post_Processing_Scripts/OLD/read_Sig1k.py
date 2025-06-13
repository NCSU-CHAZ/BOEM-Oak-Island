from scipy.io import loadmat
import pandas as pd
import os
import time

start_time = time.time()


def read_Sig1k(filepath, save_dir):  # Create read function
    Data = loadmat(
        filepath
    )  # Load mat oragnizes the 4 different data structures of the .mat file (Units, Config, Data, Description) as a
    # dictionary with four nested numpy arrays with dtypes as data field titles
    ADCPData = {}  # Initialize the dictionary we'll use
    Config = Data["Config"][0, 0]
    # Save BEAM coordinate velocity matrix
    VelArray = Data["Data"][0, 0]["Burst_Velocity_Beam"]
    reshaped = VelArray.reshape(VelArray.shape[0], -1)
    del VelArray
    ADCPData["Burst_VelBeam"] = pd.DataFrame(reshaped)

    # Save the correlation data matrix
    CorArray = Data["Data"][0, 0]["Burst_Correlation_Beam"]
    reshaped = CorArray.reshape(CorArray.shape[0], -1)
    ADCPData["Burst_CorBeam"] = pd.DataFrame(reshaped)
    del CorArray, reshaped

    # Save other fields

    ADCPData["Burst_Time"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Time"])
    ADCPData["Burst_NCells"] = pd.DataFrame(Data["Data"][0, 0]["Burst_NCells"])
    ADCPData["Burst_Pressure"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Pressure"])
    ADCPData["Burst_Heading"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Heading"])
    ADCPData["Burst_Pitch"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Pitch"])
    ADCPData["Burst_Roll"] = pd.DataFrame(Data["Data"][0, 0]["Burst_Roll"])

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

    print(f"Converted mat to hdr for {save_path}")




