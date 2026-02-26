import os
import sys
sys.path.insert(0,os.path.abspath('../../datatypes_and_database/'))
sys.path.insert(0,os.path.abspath('../../config_file_handling/'))
import grand_spectrum_core.admx_db_interface
import grand_spectrum_core.admx_db_setup
from grand_spectrum_core.admx_db_datatypes import PowerSpectrum
from grand_spectrum_core.admx_datatype_hdf5 import save_dataseries_to_hdf5,load_dataseries_from_hdf5
from grand_spectrum_core.config_file_handling import get_intermediate_data_file_name
import numpy as np
import h5py
import datetime
from dateutil import parser
from scipy.signal import savgol_filter
import yaml
import argparse
import pandas as pd
import struct
import json
import array
import pyfftw
from matplotlib import pyplot as plt

def resample_array(avgarray, num_samples):
    original_indices = np.linspace(0, len(avgarray) - 1, len(avgarray))
    new_indices = np.linspace(0, len(avgarray) - 1, num_samples)
    downsampled_avgarray = np.interp(new_indices, original_indices, avgarray)
    return downsampled_avgarray

# Heavily modified version of ADMX's reciever shape extraction. Result is the same but initial input requires HiRes Data, Run Definitions file
# May have been done incorrectly, looks like ADMX averages the file into a single reciever shape given the polynomial of all the runs.

def extract_receiver():
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
    folder_path = HR_folder + "/" + binned_data  # Replace with your folder's path
    file_prefix =run_definition["nibbles"][target_nibble]["file_prefix"]


    prefix = "admx"
    df["Filename_Tag"] = df["Filename_Tag"].apply(lambda x: prefix + x)

    # dir_name = "reciever_shape1B_HiRes"
    dir_name = HR_folder

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)  # Create the folder
        print(f"Folder created: {dir_name}")
    else:
        print(f"Folder already exists: {dir_name}")

    shape_arrays=[]

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        match_name =  file_name.replace('_binned.h5','').replace('admx','')
        if os.path.isfile(file_path):
            if df["Filename_Tag"].str.contains(match_name).any():
                # Extract the value from the 'Digitizer_Log_ID' column for the matching row
                fstart = df.loc[df["Filename_Tag"].str.contains(match_name), "Start_Frequency"].iloc[0]*1e6
                fstop = df.loc[df["Filename_Tag"].str.contains(match_name), "Stop_Frequency"].iloc[0]*1e6
            else:

                continue

            with h5py.File(file_path, "r") as file:
        # List all groups (datasets and groups) in the file
                group_name = str(list(file.keys())[0])

                if group_name in file:
                    yvalues = file[group_name]

                attrs = dict(file[group_name].attrs)
                shape = yvalues.shape
                dtype = yvalues.dtype
                xstart = attrs['fstart']
                xstop = attrs['fstop']
                step = (xstop - xstart)/len(yvalues)
                yvalues = list(yvalues)


            # with open(file_path, 'rb') as file:

            # #go to the end of the file and ask where we are to measure the size of the file
            #     file.seek(0, 2)
            #     file_size = file.tell()

            #     #go back to the beginning of the file
            #     file.seek(0, 0)

            #     #get headers
            #     (header_size,) = struct.unpack('q',file.read(8))
            #     header_bytes = file.read(header_size)
            #     header_json = json.loads(header_bytes)
            #     (h1_size,) = struct.unpack('q',file.read(8))
            #     h1_bytes = file.read(h1_size)
            #     h1_json = json.loads(h1_bytes)
            #     step = h1_json['x_spacing']*1e-6

            #     #prepare to read the data. First, get the number of data points
            #     (data_size,) = struct.unpack('Q',file.read(8))
                
            #     #now, read the data.
            #     data = array.array('f') #declare an array of floats
            #     data.frombytes(file.read(data_size*4)) #each float is 4 bytes, so read number of data points x 4 bytes
                
            #     end_read = file.tell()
                # fftd_data = [np.abs(j)**2 for j in fftd_data]
                shape_arrays.append(yvalues)

    output_file = dir_name + "/" + file_prefix + "_receiver_shape.h5"

    avgarray= np.average(shape_arrays,0)


    xs=np.linspace(0,1,len(avgarray))
    #Fit a polynomial of degree 3 to the averaged array
    filtered=np.poly1d(np.polyfit(xs,avgarray,3))(xs)


    # filtered=savgol_filter(fftd_data,17,3)
    #filtered=avgarray


    #polynomial smoothing
    averagespectrum=PowerSpectrum(avgarray,fstart,fstop)
    filteredbackground=PowerSpectrum(filtered,fstart,fstop)

    receiver_shape_file=h5py.File(output_file,"w")
    save_dataseries_to_hdf5(receiver_shape_file,"receiver_shape",filteredbackground)
    save_dataseries_to_hdf5(receiver_shape_file,"averaged_data",averagespectrum)
    receiver_shape_file.close()

def extract_receiver_hr(num_bins = 0):
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
    folder_path = HR_folder + "/" + data_folder  # Replace with your folder's path
    file_prefix =run_definition["nibbles"][target_nibble]["file_prefix"]


    prefix = "admx"
    df["Filename_Tag"] = df["Filename_Tag"].apply(lambda x: prefix + x)

    # dir_name = "reciever_shape1B_HiRes"
    dir_name = HR_folder

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)  # Create the folder
        print(f"Folder created: {dir_name}")
    else:
        print(f"Folder already exists: {dir_name}")

    shape_arrays=[]

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        match_name =  file_name.replace('admx','').replace('.dat','')
        if os.path.isfile(file_path):
            if df["Filename_Tag"].str.contains(match_name).any():
                # Extract the value from the 'Digitizer_Log_ID' column for the matching row
                fstart = df.loc[df["Filename_Tag"].str.contains(match_name), "Start_Frequency"].iloc[0]*1e6
                fstop = df.loc[df["Filename_Tag"].str.contains(match_name), "Stop_Frequency"].iloc[0]*1e6
            else:

                continue

            with open(file_path, 'rb') as file:
                file.seek(0, 2)  # Move to the end of file
                file_size = file.tell()
                file.seek(0, 0)  # Reset to the beginning

                # Read headers
                header_size = struct.unpack('q', file.read(8))[0]
                header_bytes = file.read(header_size)
                header_json = json.loads(header_bytes)

                h1_size = struct.unpack('q', file.read(8))[0]
                h1_bytes = file.read(h1_size)
                h1_json = json.loads(h1_bytes)
                time_step = h1_json['x_spacing'] * 1e-6  # Convert to seconds

                # Read data
                data_size = struct.unpack('Q', file.read(8))[0]
                data = array.array('f')
                data.frombytes(file.read(data_size * 4))
                yvalues = np.array(data)
                fft_values = pyfftw.interfaces.numpy_fft.rfft(yvalues)
                freqs = np.fft.rfftfreq(len(yvalues), d=time_step)
                shape_arrays.append(np.square(np.abs(fft_values)))

    output_file = dir_name + "/" + file_prefix + "_receiver_shape_hr.h5"

    avgarray= np.average(shape_arrays,0)

    if num_bins != 0:
        avgarray = resample_array(avgarray, num_bins)


    xs=np.linspace(0,1,len(avgarray))
    #Fit a polynomial of degree 3 to the averaged array
    filtered=np.poly1d(np.polyfit(xs,avgarray,3))(xs)
    print('Finished Fitting Polynomial. Saving to h5 file')

    # filtered=savgol_filter(fftd_data,17,3)
    #filtered=avgarray


    #polynomial smoothing
    averagespectrum=PowerSpectrum(avgarray,fstart,fstop)
    filteredbackground=PowerSpectrum(filtered,fstart,fstop)


    receiver_shape_file=h5py.File(output_file,"w")
    save_dataseries_to_hdf5(receiver_shape_file,"receiver_shape",filteredbackground)
    save_dataseries_to_hdf5(receiver_shape_file,"averaged_data",averagespectrum)
    receiver_shape_file.close()
    print('Done \n')


# Either Use:
    # extract_receiver_hr()
    # or 
    # extract_receiver()
num_bins = 2**14
