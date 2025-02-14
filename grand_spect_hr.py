import os
import sys
import math
import numpy as np
import datetime
from dateutil import parser
import yaml
from tqdm import tqdm
import argparse
from scipy.constants import codata
import h5py
import pandas as pd
#TODO other paths here
from config_file_handling import test_timestamp_cuts, frequency_cuts
from admx_db_datatypes import PowerMeasurement,ADMXDataSeries
from admx_datatype_hdf5 import save_dataseries_to_hdf5,load_dataseries_from_hdf5,save_measurement_to_hdf5,load_measurement_from_hdf5
from boosted_maxwell_boltzmann import maxwell_boltzman_shape,normalize_kernel
from turner_axion import turner_axion_shape
from lentzian import lentzian_shape
from optimal_filtering import deconvolve_with_uncertainty
from axion_parameters import ksvz_axion_power,dfsz_axion_power
from config_file_handling import get_intermediate_data_file_name
from co_addition_functions import ADMXGrandSpectrum
import pickle

default_run_definition="run1b_definitions.yaml"
target_nibble= "nibble5"

argparser=argparse.ArgumentParser()
argparser.add_argument("-r","--run_definition",help="run definition yaml file",default=default_run_definition)
argparser.add_argument("-n","--nibble_name",help="name of nibble to run",default=target_nibble)
argparser.add_argument("-c","--rescan_name",help="name of rescan",action="append")
argparser.add_argument("-a","--all_data",help="ignore scans and rescans, just dump all data",action="store_true")
argparser.add_argument("-u","--update",help="do not create a new file, just add or overwrite",action="store_true")
argparser.add_argument("--lineshape",help="one of none, maxboltz, lentz",default="maxboltz")
argparser.add_argument("--bin_resolution",help="bin resolution of grand spectrum in hz",type=float)
argparser.add_argument("--start_frequency",help="",type=float)
argparser.add_argument("--stop_frequency",help="",type=float)
argparser.add_argument("--output_file",help="override the output file")
argparser.add_argument("--only_hardware_synthetics",action="store_true")
args=argparser.parse_args()


#pull in run configuration
run_definition_file=open(args.run_definition,"r")
run_definition=yaml.load(run_definition_file, Loader = yaml.Loader)
run_definition_file.close()
target_nibble=args.nibble_name
#lineshape=args.lineshape
lineshape=run_definition["software_synthetics"]["no_injections"]["lineshape"]
print("Lineshape is ",lineshape)
processed_file_name=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"spectra.h5")
grand_start_freq=float(run_definition["nibbles"][target_nibble]["start_freq"])
grand_stop_freq=float(run_definition["nibbles"][target_nibble]["stop_freq"])
Q_mode= "transmission_q"
print('grand_start_freq')
print(grand_start_freq)
grand_bin_resolution=run_definition["nibbles"][target_nibble]["grand_resolution_hz"]
grand_file_name=get_intermediate_data_file_name(run_definition["nibbles"][args.nibble_name],"grand.h5")
grand_file_name_rescan=get_intermediate_data_file_name(run_definition["nibbles"][args.nibble_name],"grand_rescan.h5")
grand_file_name_combined=get_intermediate_data_file_name(run_definition["nibbles"][args.nibble_name],"grand_combined.h5")

if args.bin_resolution:
    print("overriding config file resolution of {} Hz with {} Hz".format(grand_bin_resolution,args.bin_resolution))
    grand_bin_resolution=args.bin_resolution
if args.start_frequency:
    print("overriding config file start frequency of {} Hz with {} Hz".format(grand_start_freq,args.start_frequency))
    grand_start_freq=args.start_frequency
if args.stop_frequency:
    print("overriding config file stop frequency of {} Hz with {} Hz".format(grand_stop_freq,args.stop_frequency))
    grand_stop_freq=args.stop_frequency
if args.output_file:
    print("overriding config file output file {}  with {}".format(grand_file_name,args.output_file))
    grand_file_name_rescan=args.output_file
    grand_file_name=args.output_file

rescan_mode=False


if args.rescan_name:
    rescan_mode=True
    print("RESCAN MODE")

def lorentz(x,x0,Q):
    return 1/(4*(x-x0)*(x-x0)*Q*Q/(x0*x0)+1)


#load nibble timestamp cuts
rescan_timestamp_cuts=[]

if "rescans" in run_definition["nibbles"][target_nibble]:
    #print(run_definition["nibbles"][target_nibble]["rescans"])
    rescan_mode=True
    for elem in run_definition["nibbles"][target_nibble]["rescans"]:
        #print(elem)
        #print(elem["why"])
        #print(run_definition["nibbles"][target_nibble]["rescans"][elem])
        #print("reading rescan {}".format(elem["why"]))
        print("reading rescan {}".format(run_definition["nibbles"][target_nibble]["rescans"][elem]["why"]))
        #rescan_timestamp_cuts.append(elem)
        #cut_def["start_time"]
        #rescan_timestamp_cuts.append(elem["start_time"], elem["stop_time"])  
        rescan_timestamp_cuts.append(run_definition["nibbles"][target_nibble]["rescans"][elem])

frequency_cut_yaml=run_definition["frequency_cuts"]

parameter_file=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"2018_05_19.txt")
useful_parameters=["Start_Frequency", "Stop_Frequency", "Digitizer_Log_ID","Integration_Time","Filename_Tag","Quality_Factor","Cavity_Resonant_Frequency","JPA_SNR","Thfet","Attenuation","Reflection","Transmission?"]
df = pd.read_csv(parameter_file, delimiter = '\t', names=useful_parameters, header=None)
prefix = "admx"
df["Filename_Tag"] = df["Filename_Tag"].apply(lambda x: prefix + x)


