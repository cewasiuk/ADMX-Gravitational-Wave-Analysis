import numpy as np 
import os 
import struct
import json
import array
import csv
from matplotlib import pyplot as plt
from admx_db_datatypes import ADMXDataSeries 
from admx_datatype_hdf5 import save_dataseries_to_hdf5
import h5py
import sys
sys.path.insert(0,os.path.abspath('../../datatypes_and_database/'))
sys.path.insert(0,os.path.abspath('../../config_file_handling/'))
import admx_db_interface
import admx_db_setup
from admx_db_datatypes import PowerSpectrum
from admx_datatype_hdf5 import save_dataseries_to_hdf5,load_dataseries_from_hdf5
from config_file_handling import get_intermediate_data_file_name
import numpy as np
import h5py
import datetime
from dateutil import parser
from scipy.signal import savgol_filter
import yaml
import argparse
import pandas as pd
import pyfftw
from matplotlib import pyplot as plt

default_run_definition="run1b_definitions.yaml"
target_nibble= "nibble5"

argparser=argparse.ArgumentParser()
argparser.add_argument("-r","--run_definition",help="run definition yaml file",default=default_run_definition)
argparser.add_argument("-n","--nibble_name",help="name of nibble to run",default=target_nibble)
argparser.add_argument("-u","--update",help="update this file (not yet implemented)",action="store_true")
argparser.add_argument("-s","--synthetic",help="synthetic signal file")
args=argparser.parse_args()


#pull in run configuration
run_definition_file=open(args.run_definition,"r")
run_definition=yaml.load(run_definition_file, Loader = yaml.Loader)
run_definition_file.close()
target_nibble=args.nibble_name

parameter_file=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"2018_05_19.txt")
useful_parameters=["Start_Frequency", "Stop_Frequency", "Digitizer_Log_ID","Integration_Time","Filename_Tag","Quality_Factor","Cavity_Resonant_Frequency","JPA_SNR","Thfet","Attenuation","Reflection","Transmission?"]
df = pd.read_csv(parameter_file, delimiter = '\t', names=useful_parameters, header=None)

HR_folder = "hr_data"
data_folder = "2018_05_19"
binned_data = 'binned_hr_data'
folder_path = HR_folder + "/" + data_folder
output_folder = "binned_hr_data"

num_bins = 2**12 
print("Number of bins is: {} \n".format(num_bins))

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    match_name =  file_name.replace('admx','')
    if os.path.isfile(file_path):
        if df["Filename_Tag"].str.contains(match_name).any():
            # Extract the value from the 'Digitizer_Log_ID' column for the matching row
            fstart = df.loc[df["Filename_Tag"].str.contains(match_name), "Start_Frequency"].iloc[0]*1e6
            fstop = df.loc[df["Filename_Tag"].str.contains(match_name), "Stop_Frequency"].iloc[0]*1e6
        else:

            continue
        data_file = str(file_name)
        data_name = data_file.replace(".dat","")



        output_file_name = HR_folder + "/" + output_folder + "/" + data_name + "_binned.h5"

        print('%%%%%%%%%%%%%%%%%%%%%% {} %%%%%%%%%%%%%%%%%%%%%%%%%%%%'.format(output_file_name))




        file_path = os.path.join(folder_path, data_file)

        with open(file_path, 'rb') as file:

            file.seek(0, 2)
            file_size = file.tell()

            #go back to the beginning of the file
            file.seek(0, 0)

            #get headers
            (header_size,) = struct.unpack('q',file.read(8))
            header_bytes = file.read(header_size)
            header_json = json.loads(header_bytes)
            (h1_size,) = struct.unpack('q',file.read(8))
            h1_bytes = file.read(h1_size)
            h1_json = json.loads(h1_bytes)
            time_step = h1_json['x_spacing']*1e-6 # in seconds

            #prepare to read the data. First, get the number of data points
            (data_size,) = struct.unpack('Q',file.read(8))
            
            #now, read the data.
            data = array.array('f')  # Declare an array of floats
            data.frombytes(file.read(data_size * 4))  # Read float data

        # Convert to NumPy array
        yvalues = np.array(data)

        # **Step 1: Compute FFT to Convert to Frequency Domain**
        fft_values = pyfftw.interfaces.numpy_fft.rfft(yvalues)
        freqs = np.fft.rfftfreq(len(yvalues), d=time_step)
        fft_values = [np.square(np.abs(i)) for i in fft_values ]
        # **Step 2: Bin the Frequency Data**
        bin_edges = np.linspace(freqs[0], freqs[-1], num_bins + 1)
        bin_indices = np.digitize(freqs, bin_edges)  # Assign each frequency to a bin

        # Compute bin averages
        bin_averages = np.zeros(num_bins)
        bin_counts = np.zeros(num_bins)

        for i in range(len(freqs)):
            bin_idx = bin_indices[i] - 1  # Convert 1-indexed to 0-indexed
            if 0 <= bin_idx < num_bins:  # Ensure it's a valid bin
                bin_averages[bin_idx] += np.abs(fft_values[i])  # Store FFT magnitude
                bin_counts[bin_idx] += 1

        # Compute bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Avoid division by zero
        bin_averages = np.divide(bin_averages, bin_counts, out=np.zeros_like(bin_averages), where=bin_counts != 0)

        # **Step 3: Save Binned Frequency Data to HDF5**
        with h5py.File(output_file_name, "w") as output_file:
            print(f"Saving FFT Binned Data for: {data_name}")

            # Create dataset and store frequency-domain values
            dataset = output_file.create_dataset(data_name, data=bin_averages)

            # Add attributes (metadata)
            dataset.attrs["fstart"] = bin_centers[0]
            dataset.attrs["fstop"] = bin_centers[-1]
            dataset.attrs["xunits"] = "Hz"
            dataset.attrs["yunits"] = "Magnitude"

            print("Attributes saved successfully!")

        print(f"FFT processing and binning completed for: {data_name}")
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%' + '\n')

print('\n \n Completed Binning')

