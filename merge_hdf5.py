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
#np.set_printoptions(threshold=sys.maxsize)
sys.path.insert(0,os.path.abspath('../../datatypes_and_database/'))
sys.path.insert(0,os.path.abspath('../../background_subtraction/sg_background_fit'))
sys.path.insert(0,os.path.abspath('../../signal_shapes/distributions'))
sys.path.insert(0,os.path.abspath('../../signal_shapes/optimal_filtering'))
sys.path.insert(0,os.path.abspath('../../signal_shapes/axion_parameters'))
sys.path.insert(0,os.path.abspath('../../config_file_handling/'))
#TODO other paths here
from config_file_handling import test_timestamp_cuts, frequency_cuts
from admx_db_datatypes import PowerSpectrum,PowerMeasurement,ADMXDataSeries
from admx_datatype_hdf5 import save_dataseries_to_hdf5,load_dataseries_from_hdf5,save_measurement_to_hdf5,load_measurement_from_hdf5
from boosted_maxwell_boltzmann import maxwell_boltzman_shape,normalize_kernel
from turner_axion import turner_axion_shape
from lentzian import lentzian_shape
from optimal_filtering import deconvolve_with_uncertainty
from axion_parameters import ksvz_axion_power,dfsz_axion_power
from config_file_handling import get_intermediate_data_file_name
from co_addition_functions import ADMXGrandSpectrum
import pickle

#default run config file
default_run_definition="../../config/run1c_definitions.yaml"
target_nibble="test_nibble"

#Parse command line arguments
argparser=argparse.ArgumentParser()
argparser.add_argument("-r","--run_definition",help="run definition yaml file",default=default_run_definition)
argparser.add_argument("-n","--nibble_name",help="name of nibble to run",default=target_nibble)
argparser.add_argument("-ns","--nibble_names_to_add",help="name of nibble to add, like nibble1b,nibble1c",default=target_nibble)
argparser.add_argument("-c","--rescan_name",help="name of rescan",action="append")
argparser.add_argument("-a","--all_data",help="ignore scans and rescans, just dump all data",action="store_true")
argparser.add_argument("-u","--update",help="do not create a new file, just add or overwrite",action="store_true")
argparser.add_argument("--merge_spectra",help="",action='store_true')
argparser.add_argument("--lineshape",help="one of none, maxboltz, lentz",default="maxboltz")
argparser.add_argument("--bin_resolution",help="bin resolution of grand spectrum in hz",type=float)
argparser.add_argument("--start_frequency",help="",type=float)
argparser.add_argument("--stop_frequency",help="",type=float)
argparser.add_argument("--output_file",help="override the output file")
argparser.add_argument("--only_hardware_synthetics",action="store_true")
args=argparser.parse_args()

nibbles_to_add=args.nibble_names_to_add.split(",")

#pull in run configuration
run_definition_file=open(args.run_definition,"r")
run_definition=yaml.load(run_definition_file)
run_definition_file.close()
target_nibble=args.nibble_name

newnibblename = target_nibble
for the_nibble in nibbles_to_add:
    newnibblename += "_"+the_nibble

#lineshape=args.lineshape
lineshape=run_definition["nibbles"][target_nibble]["lineshape"]
print("Lineshape is ",lineshape)

print("Merging grand.h5")
grand_start_freq=float(run_definition["nibbles"][target_nibble]["start_freq"])
grand_stop_freq=float(run_definition["nibbles"][target_nibble]["stop_freq"])
grand_bin_resolution=run_definition["nibbles"][target_nibble]["grand_resolution_hz"]
grand_file_names=[get_intermediate_data_file_name(run_definition["nibbles"][args.nibble_name],"grand.h5")]
for the_nibble in nibbles_to_add:
    grand_file_names.append(get_intermediate_data_file_name(run_definition["nibbles"][the_nibble],"grand.h5"))

grand_spectrum_nbins=int((grand_stop_freq-grand_start_freq)/grand_bin_resolution)
grand_spectrum=ADMXGrandSpectrum(grand_spectrum_nbins,grand_start_freq,grand_stop_freq)
grand_spectrum_postcavity=ADMXGrandSpectrum(grand_spectrum_nbins,grand_start_freq,grand_stop_freq)

fins={}
pickles={}
filtered_measurements={}
nonlorentzian_filtered_measurements={}
for the_name in grand_file_names:
    fins[the_name] = open(the_name.replace('grand.h5','filtered_measurements.pickle'), 'rb')
    pickles[the_name] = pickle.load(fins[the_name])
    filtered_measurements[the_name] = pickles[the_name][0]
    nonlorentzian_filtered_measurements[the_name] = pickles[the_name][1]
    for filtered_measurement in filtered_measurements[the_name]:
        grand_spectrum.update_with_additional_measurement(filtered_measurement)
    for nonlorentzian_filtered_measurement in nonlorentzian_filtered_measurements[the_name]:
        grand_spectrum_postcavity.update_with_additional_measurement(nonlorentzian_filtered_measurement)

avg_spectrum_axion_search=grand_spectrum.get_average_power_spectrum()
avg_spectrum_axion_search_nonlorentz=grand_spectrum_postcavity.get_average_power_spectrum()

output_file=h5py.File(grand_file_names[0].replace(target_nibble,newnibblename),"w")
grand_name="grand_spectrum"
if lineshape!="maxboltz":
    grand_name="grand_spectrum_"+lineshape
save_measurement_to_hdf5(output_file,grand_name,avg_spectrum_axion_search)
save_measurement_to_hdf5(output_file,grand_name+"_nonlorentz",avg_spectrum_axion_search_nonlorentz)
save_dataseries_to_hdf5(output_file,"axion_search_chisq",grand_spectrum.get_chisq_spectrum())
save_dataseries_to_hdf5(output_file,"axion_search_nonlorentz_chisq",grand_spectrum_postcavity.get_chisq_spectrum())
save_dataseries_to_hdf5(output_file,"contributing_spectra",grand_spectrum.get_contributing_counts())
#for quick plotting
dfsz_power_array=np.array([1/dfsz_axion_power(1,f,1) for f in avg_spectrum_axion_search.get_xvalues() ])

grand_spectrum_dfsz=avg_spectrum_axion_search * dfsz_power_array
snr_array=1.0/grand_spectrum_dfsz.yuncertainties
#350 Hz to 100 kHz
downsample_count=300
snr_array_downsampled=[ np.min(snr_array[downsample_count*i:downsample_count*(i+1)]) for i in range(math.floor(len(grand_spectrum_dfsz)/downsample_count)) ]
snr_spectrum=ADMXDataSeries(snr_array_downsampled,avg_spectrum_axion_search.xstart,avg_spectrum_axion_search.xstop)
snr_spectrum_fine=ADMXDataSeries(1.0/grand_spectrum_dfsz.yuncertainties,avg_spectrum_axion_search.xstart,avg_spectrum_axion_search.xstop)

save_dataseries_to_hdf5(output_file,"snr_vs_frequency",snr_spectrum)
save_dataseries_to_hdf5(output_file,"snr_vs_frequency_fine",snr_spectrum_fine)
output_file.close()


## Merge parameters.h5

print("Merging parameters.h5")
parameters_file_names=[get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"parameters.h5")]
for the_nibble in nibbles_to_add:
    parameters_file_names.append(get_intermediate_data_file_name(run_definition["nibbles"][the_nibble],"parameters.h5"))
new_parameters_file_name=parameters_file_names[0].replace(target_nibble,newnibblename)
output_file=h5py.File(parameters_file_names[0].replace(target_nibble,newnibblename),"w")


parameters_files={}
params = {}
for ii,parameters_file_name in enumerate(parameters_file_names):
    useful_parameters=["scanid","Q","Q_refl","Q_trans","f0","cut_reason","coupling","Tsys","ff_v","B"]
    parameters_files[parameters_file_name] = h5py.File(parameters_file_name,"r")
    for x in useful_parameters:
        dset=parameters_files[parameters_file_name][x]
        tmp_params=dset[...]

        if ii == 0:
            params[x] = tmp_params
        else:
            params[x] = np.concatenate([params[x],tmp_params])

    parameters_files[parameters_file_name].close()

stringtype = h5py.special_dtype(vlen=bytes)
dsets = {}
for key in params.keys():
    if key == "scanid":
        dsets[key] = output_file.create_dataset(key,(len(params[key]),),dtype="S25")
        dsets[key][...] = params[key]
    elif key == "cut_reason":
        dsets[key] = output_file.create_dataset(key,(len(params[key]),),dtype=stringtype)
        dsets[key][...] = params[key]
    else:
        dsets[key] = output_file.create_dataset(key,data=params[key])

if args.merge_spectra: 
    ## Merge spectra.h5
    print("Merging spectra.h5")
    spectra_file_names=[get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"spectra.h5")]
    for the_nibble in nibbles_to_add:
        spectra_file_names.append(get_intermediate_data_file_name(run_definition["nibbles"][the_nibble],"spectra.h5"))
    output_file=h5py.File(spectra_file_names[0].replace(target_nibble,newnibblename),"w")

    spectra_files={}
    scanid_array=[]
    cut_reason_array=[]
    stdev_increase_array=[]
    for spectra_file_name in spectra_file_names:
        spectra_files[spectra_file_name] = h5py.File(spectra_file_name,"r")
        for scanid in spectra_files[spectra_file_name]['scanid']:
            filtered_name = scanid.decode('UTF-8')+"/measurement_filtered"
            target_name = scanid.decode('UTF-8')+"/measurement_scaled"
            raw_name=scanid.decode('UTF-8')+"/raw_spectrum"
            yvalues = spectra_files[spectra_file_name][scanid]['measurement_filtered']['yvalues']
            yuncertainties = spectra_files[spectra_file_name][scanid]['measurement_filtered']['yuncertainties']
            measurement_filtered = PowerMeasurement(yvalues,yuncertainties,0,1)   #start and stop frequencies are not recoreded
            measurement_filtered.metadata["timestamp"]=scanid
            yvalues = spectra_files[spectra_file_name][scanid]['measurement_scaled']['yvalues']
            yuncertainties = spectra_files[spectra_file_name][scanid]['measurement_scaled']['yuncertainties']
            measurement_scaled = PowerMeasurement(yvalues,yuncertainties,0,1)        
            measurement_scaled.metadata["timestamp"]=scanid
            yvalues = spectra_files[spectra_file_name][scanid]['raw_spectrum']['yvalues']
            spectrum_raw=PowerSpectrum(yvalues,0,1)
            save_measurement_to_hdf5(output_file,filtered_name,measurement_filtered)
            save_measurement_to_hdf5(output_file,target_name,measurement_scaled)
            save_dataseries_to_hdf5(output_file,raw_name,spectrum_raw)

        scanid_array += spectra_files[spectra_file_name]['scanid']
        cut_reason_array += spectra_files[spectra_file_name]['cut_reason']
        stdev_increase_array += spectra_files[spectra_file_name]['stdev_increase']
        # spectra_files[spectra_file_name].close()

    dset_scanids=output_file.create_dataset("scanid",(len(scanid_array),),dtype="S25")
    dset_scanids[...]=scanid_array
    stringtype = h5py.special_dtype(vlen=bytes)
    dset_cutreason=output_file.create_dataset("cut_reason",(len(cut_reason_array),),dtype=stringtype)
    dset_cutreason[...]=cut_reason_array
    output_file.close()


