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

#default run config file
default_run_definition="../../config/run1c_definitions.yaml"
target_nibble="test_nibble"

#Parse command line arguments
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
run_definition=yaml.load(run_definition_file)
run_definition_file.close()
target_nibble=args.nibble_name
#lineshape=args.lineshape
lineshape=run_definition["nibbles"][target_nibble]["lineshape"]
print("Lineshape is ",lineshape)
processed_file_name=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"spectra.h5")

grand_start_freq=float(run_definition["nibbles"][target_nibble]["start_freq"])
grand_stop_freq=float(run_definition["nibbles"][target_nibble]["stop_freq"])
Q_mode=run_definition["nibbles"][target_nibble]["Q_mode"]
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

#extract the parameters I need
#parameter_file=run_definition["nibbles"][target_nibble]["parameter_file_name"]
parameter_file=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"parameters.h5")
#load the parameters
useful_parameters=["scanid","Q","Q_refl","Q_trans","f0","cut_reason","coupling","ff_v","Tsys","B"]
params={} #dictionary of the parameter arrays
f=h5py.File(parameter_file,"r")
for x in useful_parameters:
    dset=f[x]
    params[x]=dset[...]
f.close()



#load the processed spectra
processed_file=h5py.File(processed_file_name,"r")
scanids=processed_file["scanid"][...]
cut_reasons=processed_file["cut_reason"][...]

#create the empty grand spectrum and info arrays
#create the grand spectrum array, empty
grand_spectrum_nbins=int((grand_stop_freq-grand_start_freq)/grand_bin_resolution)
print("nbins are",grand_spectrum_nbins)
#power measurement


grand_spectrum=ADMXGrandSpectrum(grand_spectrum_nbins,grand_start_freq,grand_stop_freq)
grand_spectrum_postcavity=ADMXGrandSpectrum(grand_spectrum_nbins,grand_start_freq,grand_stop_freq)
if rescan_mode:
   grand_spectrum_rescan=ADMXGrandSpectrum(grand_spectrum_nbins,grand_start_freq, grand_stop_freq)
   grand_spectrum_rescan_postcavity=ADMXGrandSpectrum(grand_spectrum_nbins,grand_start_freq, grand_stop_freq)
   grand_spectrum_combined=ADMXGrandSpectrum(grand_spectrum_nbins,grand_start_freq, grand_stop_freq)
   grand_spectrum_combined_postcavity=ADMXGrandSpectrum(grand_spectrum_nbins,grand_start_freq, grand_stop_freq)


#average Q of spectra contributing
q_sum_array=np.zeros(grand_spectrum_nbins)
#average 1/T^2 of spectra contributing
inv_temp_squared_array=np.zeros(grand_spectrum_nbins)

filtered_measurements=[]
nonlorentzian_filtered_measurements=[]

first_pass=True
#loop through all of the processed spectra
print("AHHHH")
print(len(scanids))

for i in tqdm(range(len(scanids))):
    scanid=scanids[i]
    
    if scanid not in params["scanid"]:
        print("Error parameters don't exist for scan "+scanid.decode('UTF-8'))
        continue
    if cut_reasons[i].decode('UTF-8')!="":
        print('cut reason')
        continue
    cut_reason=""
    timestamp=parser.parse(scanid.decode('UTF-8'))
    #TODO this is a mess, redo it sometime
    #for cut_def in rescan_timestamp_cuts:
    #  print(cut_def["start_time"])
    #print(timestamp)
    should_cut_rescan,cut_reason= test_timestamp_cuts(rescan_timestamp_cuts,timestamp)
    #should_cut_rescan,cut_rescan= frequency_cuts(rescan_timestamp_cuts, timestamp, scanid)
    #if (not args.all_data) and should_cut_rescan and not rescan_mode: #if we're in a rescan and not in rescan mode, cut it
        #print("cutting due to rescan")
        #continue
        
    #if (not args.all_data) and not should_cut_rescan and rescan_mode: #if we're not in a rescan, and in rescan mode, cut it
        #continue

    ind=np.where(params["scanid"]==scanid)[0][0]
    spectrum_loc=scanid.decode('UTF-8')+"/measurement_scaled"
    if spectrum_loc not in processed_file:
        print("warning, couldn't find "+spectrum_loc+" in file, skipping")
        continue

    measurement=load_measurement_from_hdf5(processed_file,spectrum_loc)


    if args.only_hardware_synthetics: #in this mode, only keep things with hardware syntehtics
        if ("synthetic_contamination" not in measurement.metadata ) or measurement.metadata["synthetic_contamination"]!="True":
            print('arg')
            continue


    if measurement.xstop<grand_spectrum.start_freq or measurement.xstart>grand_spectrum.stop_freq:
        print('are we out of range?')
        continue #outside our range, ignore
    #so the measurement comes in Watts per Kelvin
    Tsys=params["Tsys"][ind]
    ff_v=params["ff_v"][ind]
    Q=params["Q"][ind]
    Q_refl=params["Q_refl"][ind]
    Q_trans=params["Q_trans"][ind]
    B=params["B"][ind]
    f0=params["f0"][ind]
    #scale to unit experiment
    if Q_mode=='reflection_q':
        measurement_power=measurement*(Tsys/(ff_v*Q_refl*B*B))
    elif Q_mode=='transmission_q':
        measurement_power=measurement*(Tsys/(ff_v*Q_trans*B*B))
    else:
        measurement_power=measurement*(Tsys/(ff_v*Q_trans*B*B))


    # TODO bring back lentzian option
    kernel=None
    if lineshape=="maxboltz":
        #apply filter to spectrum
        kernel_nbins=30
        #TODO figure out the right number for this
        binwidth_hz=measurement.get_xspacing()

        #The 1.7 comes from the HAYSTAC paper
        Eavg_hz=measurement.xstart*math.pow(270.0/3e5,2)*1.7/3.0 #TODO this shouldn't be hard coded in
        kernel=maxwell_boltzman_shape(kernel_nbins,binwidth_hz,Eavg_hz)
    if lineshape=="lentz":
        kernel_nbins=30
        kernel=normalize_kernel(lentzian_shape(kernel_nbins,measurement.xstart,measurement.get_xspacing()))
    if lineshape=="turner":
        #print("Using turner lineshape")
        kernel_nbins=30
        binwidth_hz=measurement.get_xspacing()
        Eavg_hz=measurement.xstart
        v_halo=run_definition["lineshape_parameters"]["v_halo"]
        v_sun=run_definition["lineshape_parameters"]["v_sun"]
        v_earth=run_definition["lineshape_parameters"]["v_earth"]
        v_rot=run_definition["lineshape_parameters"]["v_rot"]
        parallel=run_definition["lineshape_parameters"]["parallel"]
        alpha=run_definition["lineshape_parameters"]["alpha"]
        psi=run_definition["lineshape_parameters"]["psi"]
        omega_o=run_definition["lineshape_parameters"]["omega_o"]
        omega_r=run_definition["lineshape_parameters"]["omega_r"]
        parameter_names=["v_halo","v_sun","v_earth","v_rot","parallel","alpha","psi","omega_o","omega_r"]
        parameter_values=[v_halo,v_sun,v_earth,v_rot,parallel,alpha,psi,omega_o,omega_r]
        turner_parameters=dict(zip(parameter_names,parameter_values))
        # Full Turner distribution accounts for annual and diurnal modulation as well
        scan_time = datetime.datetime.fromtimestamp(timestamp.timestamp())
        # Unix timestamp for June 2 at midnight, 2017
        midnight_june_2 = 1496361600
        start_time = datetime.datetime.fromtimestamp(midnight_june_2)
        elapsed_time = scan_time-start_time
        kernel=turner_axion_shape(kernel_nbins,binwidth_hz,Eavg_hz,elapsed_time.total_seconds(),turner_parameters)
    if lineshape!="none":
        filtered_measurement=deconvolve_with_uncertainty(measurement_power,kernel)
    else:
        filtered_measurement=measurement_power

    #if first pass, record debug info
    if first_pass:
        first_pass=False

    if Q_mode=='reflection_q':
       lorentzian_array=np.array([lorentz(f,f0,Q_refl) for f in filtered_measurement.get_xvalues()])
    elif Q_mode=='transmission_q':
       lorentzian_array=np.array([lorentz(f,f0,Q_trans) for f in filtered_measurement.get_xvalues()])
    else:
       lorentzian_array=np.array([lorentz(f,f0,Q) for f in filtered_measurement.get_xvalues()])
    nonlorentzian_filtered_measurement=filtered_measurement*lorentzian_array

    filtered_measurements.append(filtered_measurement)
    nonlorentzian_filtered_measurements.append(nonlorentzian_filtered_measurement)

    #merge into one of the grand spectra
    if (not args.all_data) and should_cut_rescan and rescan_mode: #if we're in a rescan add it to the rescan grand spectrum
        print('what about here')
        grand_spectrum_rescan.update_with_additional_measurement(filtered_measurement)
        grand_spectrum_rescan_postcavity.update_with_additional_measurement(nonlorentzian_filtered_measurement)

    else: #otherwise add it to the main grand spectrum
       print('is it getting here?')
       grand_spectrum.update_with_additional_measurement(filtered_measurement)  
       grand_spectrum_postcavity.update_with_additional_measurement(nonlorentzian_filtered_measurement)
    if rescan_mode: 
       grand_spectrum_combined.update_with_additional_measurement(filtered_measurement)
       grand_spectrum_combined_postcavity.update_with_additional_measurement(nonlorentzian_filtered_measurement)



avg_spectrum_axion_search=grand_spectrum.get_average_power_spectrum()
avg_spectrum_axion_search_nonlorentz=grand_spectrum_postcavity.get_average_power_spectrum()

#histogram of deviations
snr_bins=np.linspace(-6,6,100)
snr_scatter_histogram=np.bincount(np.digitize(avg_spectrum_axion_search.yvalues/avg_spectrum_axion_search.yuncertainties,snr_bins),minlength=len(snr_bins)+1)


#save to disk
the_file_name=grand_file_name
#if rescan_mode:
#    the_file_name=grand_file_name_rescan
if args.update:
    output_file=h5py.File(the_file_name,"a")
else:
    output_file=h5py.File(the_file_name,"w")
kernel_name="kernel"
grand_name="grand_spectrum"
snr_dev_name="snr_deviation_histogram"
combined_name="combined_grand_spectrum"
if lineshape!="maxboltz":
    kernel_name="kernel_"+lineshape
    grand_name="grand_spectrum_"+lineshape
    snr_dev_name="snr_deviation_"+lineshape
    combined_name="combined_grand_spectrum_"+lineshape

save_measurement_to_hdf5(output_file,grand_name,avg_spectrum_axion_search)
save_measurement_to_hdf5(output_file,grand_name+"_nonlorentz",avg_spectrum_axion_search_nonlorentz)
save_dataseries_to_hdf5(output_file,"axion_search_chisq",grand_spectrum.get_chisq_spectrum())
save_dataseries_to_hdf5(output_file,"axion_search_nonlorentz_chisq",grand_spectrum_postcavity.get_chisq_spectrum())
save_dataseries_to_hdf5(output_file,"contributing_spectra",grand_spectrum.get_contributing_counts())
save_dataseries_to_hdf5(output_file,snr_dev_name,ADMXDataSeries(snr_scatter_histogram,snr_bins[0],snr_bins[-1]))

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


with open(grand_file_name.replace('grand.h5','filtered_measurements.pickle'), 'wb') as f:
    pickle.dump((filtered_measurements,nonlorentzian_filtered_measurements), f)

if rescan_mode: #save rescan data if we were in rescan mode
   avg_spectrum_axion_search=grand_spectrum_rescan.get_average_power_spectrum()
   avg_spectrum_axion_search_nonlorentz=grand_spectrum_rescan_postcavity.get_average_power_spectrum()
   
   #histogram of deviations
   snr_bins=np.linspace(-6,6,100)
   snr_scatter_histogram=np.bincount(np.digitize(avg_spectrum_axion_search.yvalues/avg_spectrum_axion_search.yuncertainties,snr_bins),minlength=len(snr_bins)+1)
   the_file_name=grand_file_name_rescan
   if args.update:
    output_file=h5py.File(the_file_name,"a")
   else:
    output_file=h5py.File(the_file_name,"w")

   save_measurement_to_hdf5(output_file,grand_name,avg_spectrum_axion_search)
   save_measurement_to_hdf5(output_file,grand_name+"_nonlorentz",avg_spectrum_axion_search_nonlorentz)
   save_dataseries_to_hdf5(output_file,"axion_search_chisq",grand_spectrum_rescan.get_chisq_spectrum())
   save_dataseries_to_hdf5(output_file,"axion_search_nonlorentz_chisq",grand_spectrum_rescan_postcavity.get_chisq_spectrum())
   save_dataseries_to_hdf5(output_file,"contributing_spectra",grand_spectrum_rescan.get_contributing_counts())
   save_dataseries_to_hdf5(output_file,snr_dev_name,ADMXDataSeries(snr_scatter_histogram,snr_bins[0],snr_bins[-1]))

 

#if rescan, merge with main rescan file name, making a combined spectrum
if rescan_mode: 
    #main_grand_file=h5py.File(grand_file_name,"r")
    #print(grand_file_name)
    #print(grand_name)
    #main_grand_spectrum=load_measurement_from_hdf5(main_grand_file,grand_name)
    #main_grand_file.close()
    #print("Main Grand Spectrum xstart"+str(main_grand_spectrum.xstart))
    #print("Grand Spectrum xstart"+str(grand_spectrum.xstart))
    #combine new grand with old grand
    #grand_spectrum.update_with_additional_measurement(grand_spectrum_rescan.sum_weights)
    #grand_spectrum_postcavity.update_with_additional_measurement(grand_spectrum_rescan_postcavity.sum_weights)
    #if rescan data gives a better limit  (i.e. because of transient candidate), replace with just rescan data
    '''
    ninety_sigma=1.283
    for i in range(len(main_grand_spectrum)):
        combined_limit=main_grand_spectrum.yvalues[i]+ninety_sigma*main_grand_spectrum.yuncertainties[i]
        rescan_limit=grand_spectrum.yvalues[i]+ninety_sigma*grand_spectrum.yuncertainties[i]
        if grand_spectrum.yvalues[i]<0: #this is a more conservative (and just better) approx for negative yvalues
            rescan_limit=ninety_sigma*grand_spectrum.yuncertainties[i]
        if rescan_limit<combined_limit:
            main_grand_spectrum.yvalues[i]=grand_spectrum.yvalues[i]
            main_grand_spectrum.yuncertainties[i]=grand_spectrum.yuncertainties[i]
    '''
    the_file_name=grand_file_name_combined
    if args.update:
       output_file=h5py.File(the_file_name,"a")
    else:
       output_file=h5py.File(the_file_name,"w")
    
    avg_spectrum_axion_search=grand_spectrum_combined.get_average_power_spectrum()
    avg_spectrum_axion_search_nonlorentz=grand_spectrum_combined_postcavity.get_average_power_spectrum()
  
    #histogram of deviations
    snr_bins=np.linspace(-6,6,100)
    snr_scatter_histogram=np.bincount(np.digitize(avg_spectrum_axion_search.yvalues/avg_spectrum_axion_search.yuncertainties,snr_bins),minlength=len(snr_bins)+1)

    save_measurement_to_hdf5(output_file,grand_name,avg_spectrum_axion_search)
    save_measurement_to_hdf5(output_file,grand_name+"_nonlorentz",avg_spectrum_axion_search_nonlorentz)
    save_dataseries_to_hdf5(output_file,"axion_search_chisq",grand_spectrum_combined.get_chisq_spectrum())
    save_dataseries_to_hdf5(output_file,"axion_search_nonlorentz_chisq",grand_spectrum_combined_postcavity.get_chisq_spectrum())
    save_dataseries_to_hdf5(output_file,"contributing_spectra",grand_spectrum_combined.get_contributing_counts())
    save_dataseries_to_hdf5(output_file,snr_dev_name,ADMXDataSeries(snr_scatter_histogram,snr_bins[0],snr_bins[-1]))
    
    snr_array_downsampled=[ np.min(snr_array[downsample_count*i:downsample_count*(i+1)]) for i in range(math.floor(len(grand_spectrum_dfsz)/downsample_count)) ]
    snr_spectrum=ADMXDataSeries(snr_array_downsampled,avg_spectrum_axion_search.xstart,avg_spectrum_axion_search.xstop)
    snr_spectrum_fine=ADMXDataSeries(1.0/grand_spectrum_dfsz.yuncertainties,avg_spectrum_axion_search.xstart,avg_spectrum_axion_search.xstop)

    print(snr_array[np.where(snr_array > 1e-25)])
    save_dataseries_to_hdf5(output_file,"snr_vs_frequency",snr_spectrum)
    save_dataseries_to_hdf5(output_file,"snr_vs_frequency_fine",snr_spectrum_fine)
output_file.close()
print("Grand Spectrum Assembly Done")
