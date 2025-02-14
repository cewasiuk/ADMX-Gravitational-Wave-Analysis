import os
import sys
import math
import numpy as np
import datetime
from dateutil import parser
import yaml
import argparse
from scipy.constants import codata
import h5py
sys.path.insert(0,os.path.abspath('../../datatypes_and_database/'))
sys.path.insert(0,os.path.abspath('../../background_subtraction/sg_background_fit'))
sys.path.insert(0,os.path.abspath('../../signal_shapes/distributions'))
sys.path.insert(0,os.path.abspath('../../signal_shapes/optimal_filtering'))
sys.path.insert(0,os.path.abspath('../../signal_shapes/axion_parameters'))
sys.path.insert(0,os.path.abspath('../../config_file_handling/'))
#TODO other paths here
from config_file_handling import test_timestamp_cuts
from admx_db_datatypes import PowerMeasurement,ADMXDataSeries
from admx_datatype_hdf5 import save_dataseries_to_hdf5,load_dataseries_from_hdf5,save_measurement_to_hdf5,load_measurement_from_hdf5
from boosted_maxwell_boltzmann import maxwell_boltzman_shape
from optimal_filtering import deconvolve_with_uncertainty
from axion_parameters import ksvz_axion_power,dfsz_axion_power

#given [ [1,3,x],[2,4,y],[5,6,z] ]  return [[1,4,x>y?x:y], [5,6,;]]  from the internet
def merge_intervals(intervals):
    s = sorted(intervals, key=lambda t: t[0])
    m = 0
    for  t in s:
        if t[0] > s[m][1]:
            m += 1
            s[m] = t
        else:
            s[m] = [min(s[m][0],t[0]), max(t[1],s[m][1]), max(s[m][2],t[2])]
    return s[:m+1]


#default run config file
default_run_definition="../../config/run1b_definitions.yaml"
target_nibble="test_nibble"

#Parse command line arguments
argparser=argparse.ArgumentParser()
argparser.add_argument("-r","--run_definition",help="run definition yaml file",default=default_run_definition)
argparser.add_argument("-n","--nibble_name",help="name of nibble to run",default=target_nibble)
argparser.add_argument("-s","--min_snr",help="minimum snr",default=target_nibble,type=float)
args=argparser.parse_args()

#pull in run configuration
run_definition_file=open(args.run_definition,"r")
run_definition=yaml.load(run_definition_file)
run_definition_file.close()
target_nibble=args.nibble_name
processed_file_name=run_definition["nibbles"][target_nibble]["spectrum_file_name"]

grand_start_freq=float(run_definition["nibbles"][target_nibble]["start_freq"])
grand_stop_freq=float(run_definition["nibbles"][target_nibble]["stop_freq"])
grand_bin_resolution=run_definition["nibbles"][target_nibble]["grand_resolution_hz"]
grand_file_name=run_definition["nibbles"][target_nibble]["grand_file_name"]

#Load the grand file and snr array
input_file=h5py.File(grand_file_name,"r")
snr_spectrum=load_dataseries_from_hdf5(input_file,"snr_vs_frequency")
input_file.close()
#get all points below cutoff
span_array=[]
freqs=snr_spectrum.get_xvalues()
#my_bandwidth=700.0/60000 #roughly a half Q-width
my_bandwidth=0.025 #quarter digitizer bandwidth
for i in range(len(snr_spectrum)):
    if snr_spectrum.yvalues[i]<args.min_snr:
        span_array.append( [0.001*math.floor((freqs[i]-my_bandwidth)*1000),0.001*math.ceil((freqs[i]+my_bandwidth)*1000),1] )
#merge points into rescan regions
merged_span_array=merge_intervals(span_array)
rounded_span_array=[]
#round and merge
for i in range(len(merged_span_array)):
    rounded_span_array.append([ 0.001*math.floor(merged_span_array[i][0]*1000),0.001*math.ceil(merged_span_array[i][1]*1000),1])
merged_span_array=rounded_span_array


#print out
sum_size=0
for i in range(len(merged_span_array)):
    sum_size+=(merged_span_array[i][1]-merged_span_array[i][0])
    print(merged_span_array[i][0],merged_span_array[i][1])
print("total rescan size {} Hz".format(sum_size))
merged_rescan_array=merged_span_array

slow_rod1=400
fast_rod1=2500
frout=open("snr_rescan.lua","w")
frout.write("slow_speed={}\n".format(slow_rod1))
frout.write("fast_speed={}\n".format(fast_rod1))
frout.write("fast_int={}\n".format(2))
frout.write("slow_int={}\n".format(100))
for i in range(len(merged_rescan_array)-1):
    frout.write("if f0>{:.3f} and f0<{:.3f} then\n".format(merged_rescan_array[i][0],merged_rescan_array[i][1]))
    frout.write("    update_daq_variables( 'take_data_rod1_steps',slow_speed)\n")
    frout.write("    update_daq_variables( 'take_data_rod2_steps',slow_speed)\n")
    frout.write("    update_daq_variables( 'integration_time',slow_int)\n")
    frout.write("end\n")
    frout.write("if f0>{:.3f} and f0<{:.3f} then\n".format(merged_rescan_array[i][1],merged_rescan_array[i+1][0]))
    frout.write("    update_daq_variables( 'take_data_rod1_steps',fast_speed)\n")
    frout.write("    update_daq_variables( 'take_data_rod2_steps',fast_speed)\n")
    frout.write("    update_daq_variables( 'integration_time',fast_int)\n")
    frout.write("end\n")
frout.close()




