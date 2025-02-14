import os
from statistics import mean
import sys
import math
sys.path.insert(0,os.path.abspath('../../datatypes_and_database/'))
sys.path.insert(0,os.path.abspath('../../background_subtraction/sg_background_fit'))
sys.path.insert(0,os.path.abspath('../../background_subtraction/pade_background_fit'))
sys.path.insert(0,os.path.abspath('../../config_file_handling/'))
sys.path.insert(0,os.path.abspath('../../signal_shapes/axion_parameters'))
sys.path.insert(0,os.path.abspath('../../signal_shapes/distributions'))
import admx_db_interface
from admx_db_datatypes import PowerSpectrum,PowerMeasurement,ADMXDataSeries
from admx_datatype_hdf5 import save_dataseries_to_hdf5,load_dataseries_from_hdf5,save_measurement_to_hdf5,load_measurement_from_hdf5
from sg_background_fit import  filter_bg_with_sg,filter_bg_with_sg_keep_extrema
from pade_background_fit import filter_bg_with_pade,filter_bg_with_pade_plus_signal,filter_bg_with_predefined_filter
from pade_background_fit import mypade
from boosted_maxwell_boltzmann import maxwell_boltzman_shape
from turner_axion import turner_axion_shape
from config_file_handling import test_timestamp_cuts,frequency_cuts
from config_file_handling import get_intermediate_data_file_name
from datetime import timedelta
from collections import namedtuple
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import h5py
import datetime
from dateutil import parser
import yaml
import argparse
from scipy.constants import codata
from scipy.optimize import curve_fit
from axion_parameters import ksvz_axion_power,dfsz_axion_power
import random

kboltz=codata.value('Boltzmann constant')

#default run config file
default_run_definition="run1d_definitions.yaml"
target_nibble="test_nibble"


argparser=argparse.ArgumentParser()
argparser.add_argument("-r","--run_definition",help="run definition yaml file",default=default_run_definition)
argparser.add_argument("-n","--nibble_name",help="name of nibble to run",default=target_nibble)
argparser.add_argument("-u","--update",help="update this file (not yet implemented)",action="store_true")
argparser.add_argument("-s","--synthetic",help="synthetic signal file")
args=argparser.parse_args()

#pull in run configuration
run_definition_file=open(args.run_definition,"r")
run_definition=yaml.load(run_definition_file)
run_definition_file.close()
target_nibble=args.nibble_name

#get the required information out of the configuration file
start_time=run_definition["nibbles"][target_nibble]["start_time"]
stop_time=run_definition["nibbles"][target_nibble]["stop_time"]
bg_fit_type=run_definition["nibbles"][target_nibble]["bg_fit_type"]
Q_mode=run_definition["nibbles"][target_nibble]["Q_mode"]
#parameter_file=run_definition["nibbles"][target_nibble]["parameter_file_name"]
parameter_file=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"parameters.h5")
#receiver_shape_file=run_definition["nibbles"][target_nibble]["receiver_shape_file"]
receiver_shape_file=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"receiver_shape.h5")
#output_file_name=run_definition["nibbles"][target_nibble]["spectrum_file_name"]
output_file_name=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"spectra.h5")
timestamp_cut_yaml=run_definition["timestamp_cuts"]
stdev_max_cut=run_definition["parameter_cuts"]["max_stdev_increase"]
frequency_cut_yaml=run_definition["frequency_cuts"]
maskbins=run_definition["nibbles"][target_nibble]["maskbins"]
pade_params_output_file_name=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"pade_params.h5")

#print("{} frequency cuts loaded ".format(len(frequency_cut_yaml)))

#load the parameters
useful_parameters=["scanid","Q","Q_refl","Q_trans","f0","cut_reason","coupling","Tsys","ff_v","B"]
params={} #dictionary of the parameter arrays
f=h5py.File(parameter_file,"r")
for x in useful_parameters:
    dset=f[x]
    params[x]=dset[...]
f.close()

#load synthetic software injection

synthetic_signals=[]
if args.synthetic:
    synth_file=open(args.synthetic,"r")
    synth_data=yaml.load(synth_file)
    synthetic_signals=synth_data["signals"]
    synth_file.close()
    print("{} synthetic signals loaded".format(len(synthetic_signals)))



def lorentz(x,x0,Q):
    return 1/(4*(x-x0)*(x-x0)*Q*Q/(x0*x0)+1)

#load the background shape
f=h5py.File(receiver_shape_file,"r")
receiver_shape=load_dataseries_from_hdf5(f,"receiver_shape")
f.close()

if args.update:
    processed_file=h5py.File(output_file_name,"r")
    #TODO Update here
    scanids=processed_file["scanid"][...]
    cut_reasons=processed_file["cut_reason"][...]
    processed_file.close()

#load the scans from the database
db=admx_db_interface.ADMXDB()
db.hostname=run_definition["hostname"]
#db.hostname="admxdb01.fnal.gov"
db.dbname=run_definition["database"]
if "database_port" in run_definition:
    db.port=run_definition["database_port"]
is_sidecar=False
if "channel" in run_definition:
    if run_definition["channel"]=="sidecar":
        print("SIDECAR MODE")
        is_sidecar=True

# Turner Lineshape variables below
# Unix timestamp for June 2 at midnight, 2017
midnight_june_2 = 1496361600

v_halo=run_definition["lineshape_parameters"]["v_halo"]
v_sun=run_definition["lineshape_parameters"]["v_sun"]
v_earth=run_definition["lineshape_parameters"]["v_earth"]
v_rot=run_definition["lineshape_parameters"]["v_rot"]
parallel=run_definition["lineshape_parameters"]["parallel"]
alpha=run_definition["lineshape_parameters"]["alpha"]
psi=run_definition["lineshape_parameters"]["psi"]
omega_o=run_definition["lineshape_parameters"]["omega_o"]
omega_r=run_definition["lineshape_parameters"]["omega_r"]
turner_parameter_names=["v_halo","v_sun","v_earth","v_rot","parallel","alpha","psi","omega_o","omega_r"]
turner_parameter_values=[v_halo,v_sun,v_earth,v_rot,parallel,alpha,psi,omega_o,omega_r]
turner_parameters=dict(zip(turner_parameter_names,turner_parameter_values))

#pade filter parameters for simulating data
pade_filter_param_a=run_definition["software_synthetics"]["pade_filter_params"]["a"]
pade_filter_param_b=run_definition["software_synthetics"]["pade_filter_params"]["b"]
pade_filter_param_c=run_definition["software_synthetics"]["pade_filter_params"]["c"]
pade_filter_param_d=run_definition["software_synthetics"]["pade_filter_params"]["d"]
pade_filter_param_e=run_definition["software_synthetics"]["pade_filter_params"]["e"]
pade_filter_param_f=run_definition["software_synthetics"]["pade_filter_params"]["f"]
pade_filter_param_a_dev=run_definition["software_synthetics"]["pade_filter_params_dev"]["a_dev"]
pade_filter_param_b_dev=run_definition["software_synthetics"]["pade_filter_params_dev"]["b_dev"]
pade_filter_param_c_dev=run_definition["software_synthetics"]["pade_filter_params_dev"]["c_dev"]
pade_filter_param_d_dev=run_definition["software_synthetics"]["pade_filter_params_dev"]["d_dev"]
pade_filter_param_e_dev=run_definition["software_synthetics"]["pade_filter_params_dev"]["e_dev"]
pade_filter_param_f_dev=run_definition["software_synthetics"]["pade_filter_params_dev"]["f_dev"]

check_simulate_raw_spectra = run_definition["software_synthetics"]["simulate_raw_spectra"]
#check_simulate_raw_spectra = False #Hard code to False for now

max_lines=1000000
query=""

if is_sidecar==True:
    query="SELECT A.timestamp, B.start_frequency,B.stop_frequency,B.frequency_resolution,B.power_spectrum,B.sampling_rate,B.integration_time,A.digitizer_log_reference,A.notes from sidecar_axion_scan_log as A INNER JOIN sidecar_digitizer_log as B ON A.digitizer_log_reference=B.digitizer_log_id WHERE A.timestamp < '"+str(stop_time)+"' AND A.timestamp>'"+str(start_time)+"' ORDER BY A.timestamp desc LIMIT "+str(max_lines)
else:
    query="SELECT A.timestamp, B.start_frequency_channel_one, B.stop_frequency_channel_one, B.frequency_resolution_channel_one, B.power_spectrum_channel_one, B.sampling_rate, B.integration_time, A.digitizer_log_reference, A.notes, B.lo_frequency_channel_one, A.mode_frequency_channel_one from axion_scan_log as A INNER JOIN digitizer_log as B ON A.digitizer_log_reference=B.digitizer_log_id WHERE A.timestamp < '"+str(stop_time)+"' AND A.timestamp>'"+str(start_time)+"' AND B.notes='taking data' ORDER BY A.timestamp desc LIMIT "+str(max_lines)
print("Quering database")

records=db.send_admxdb_query(query)
print("#got "+str(len(records))+" entries")
if len(records) >= max_lines : 
    print("!!!! Data may be larger than max_lines ",max_lines,", change it bigger !!!!")
    exit()


#here's where we'll write everytihng
output_file=h5py.File(output_file_name,'w')

#arrays to store in the file
scanid_array=[]
cut_reason_array=[]
stdev_increase_array=[]

#data quality information to automatically collect
snr_bins=np.linspace(-5,5,100)
snr_scatter_histogram=np.zeros(len(snr_bins)+1)
cut_reason_histogram={}

    
#This function tells us whether this raw spectrum is scheduled for a synthetic injection
def should_inject_synthetic(measurement_filtered):
    for x in synthetic_signals:
        frequency=x["frequency"]
        if frequency>measurement_filtered.xstart and frequency < measurement_filtered.xstop:
            return True
    return False

def normalize_kernel(kernel):
    mysum=0
    for x in kernel:
        mysum+=x
    return kernel/mysum
    

#inject the signals onto raw data and refilter
def inject_synthetic(raw_spectrum,expected_ksvz_snr,filter_shape,f0,Q):
    #expected_ksvz snr is instantaneous snr, so it gets scaled by the mean
    #print("injecting! {}".format(raw_spectrum.metadata["scanid"]))
    modified_raw=raw_spectrum.copy()
    
    #Turner: Copying from the grand_spectrum_assembly
    timestamp=parser.parse(raw_spectrum.metadata["scanid"].decode('UTF-8'))
    # Full Turner distribution accounts for annual and diurnal modulation as well
    scan_time = datetime.datetime.fromtimestamp(timestamp.timestamp())
    start_time = datetime.datetime.fromtimestamp(midnight_june_2)
    elapsed_time = scan_time-start_time    

    for x in synthetic_signals:
        frequency=x["frequency"]
        if frequency<(raw_spectrum.xstart-0.01) or frequency > raw_spectrum.xstop:
            continue
        binwidth_hz=raw_spectrum.get_xspacing()
        if x["lineshape"]=="maxboltz":
            #The 1.7 comes from the HAYSTAC paper
            Eavg_hz=f0*math.pow(270.0/3e5,2)*1.7/3.0 #TODO this shouldn't be hard coded in
            kernel=normalize_kernel(maxwell_boltzman_shape(len(raw_spectrum),binwidth_hz,Eavg_hz))
        elif x["lineshape"]=="delta":
            kernel=np.zeros(len(raw_spectrum))
            kernel[0]=1.0
        elif x["lineshape"]=="turner":
            kernel=turner_axion_shape(len(raw_spectrum),binwidth_hz,f0,elapsed_time.total_seconds(),turner_parameters)
        else:
            kernel=np.zeros(len(raw_spectrum))
            kernel[0]=1.0
            print("error unknown lineshape for injection , using delta")
        power_toadd=np.zeros(len(raw_spectrum))
        start_bin=raw_spectrum.get_x_index_below_x(frequency)
        for i in range(start_bin,len(raw_spectrum)):
            r=x["ksvz_power_ratio"]
            f=raw_spectrum.get_x_at_index(i)
            power_toadd[i]=r*kernel[i-start_bin]*expected_ksvz_snr*filter_shape[i]*lorentz(f,f0,Q)
        modified_raw.yvalues+=power_toadd
    return modified_raw

#This function removes the receiver shape from the raw spectrum
#returns the fileterd spectrum and the filter used
def convert_raw_spectrum_to_filtered_spectrum(**kwargs):
    spectrum_raw=kwargs["raw_spectrum"]
    receiver_shape=kwargs["shape_receiver"]
    fit_type=kwargs["type_fit"]
        #spectrum_raw,receiver_shape,predefined_filter,fit_type="pade"):
    #Remove Warm Receiver Shape
    spectrum_before_receiver_yvalues=spectrum_raw.yvalues/receiver_shape.yvalues
    spectrum_before_receiver=PowerSpectrum(spectrum_before_receiver_yvalues, spectrum_raw.xstart, spectrum_raw.xstop)

    #Fit Residual Shap
    if fit_type=="predefined":
        predefined_filter=kwargs["filter_predefined"]
        predefined_filter_yvals=predefined_filter.yvalues/receiver_shape.yvalues
        spectrum_after_filter,filter_shape=filter_bg_with_predefined_filter(spectrum_before_receiver,predefined_filter_yvals)
        filter_shape=np.array(filter_shape)
        #This is not an option for this function so I'm just going to set them equal for now?         
        spectrum_after_filter2=spectrum_after_filter
        #filter shape has already been scaled by the receiver shape
        #return spectrum_after_filter,filter_shape*receiver_shape.yvalues,spectrum_after_filter2
    elif fit_type=="pade":
        spectrum_after_filter,filter_shape,fit_params,spectrum_after_filter2=filter_bg_with_pade(spectrum_before_receiver)
        for i in range(len(fit_params)):
            spectrum_after_filter.metadata["bg_fit_param_{}".format(i)]=fit_params[i]
    elif fit_type=="sg":
        spectrum_after_filter,filter_shape=filter_bg_with_sg_keep_extrema(spectrum_before_receiver,2,129,0.05)
    elif fit_type=="pade+sig":
        spectrum_after_filter,filter_shape,fit_params,spectrum_after_filter2=filter_bg_with_pade_plus_signal(spectrum_before_receiver)
    else:
        print("Error no fit type: ",fit_type)
        
    badness=0
    if not isinstance(filter_shape,np.ndarray):
        badness=1
       #indicates an error in filtering
        print("error in filter shape")
        return spectrum_after_filter,filter_shape,spectrum_after_filter2,badness


    ##spectrum_filtered,filter_shape=spectrum_after_filter/spectrum_before_receiver.yvalues.mean()
    return spectrum_after_filter,filter_shape*receiver_shape.yvalues,spectrum_after_filter2,badness

#This function to get the pade fit parameters
def get_pade_filter_params(spectrum_raw,receiver_shape, linecounter):
    #Remove Warm Receiver Shape
    spectrum_before_receiver=spectrum_raw/receiver_shape.yvalues

    spectrum_after_filter,filter_shape,fit_params,spectrum_after_filter2=filter_bg_with_pade(spectrum_before_receiver)
    for i in range(len(fit_params)):
        pade_fit_params_array[i][linecounter] = fit_params[i]

#Simulate a raw sub spectrum using pade filter
def get_simulated_spectrum(spectrum_start_freq, spectrum_stop_freq):

    #Simulate the pade filter parameters
    random_par_a = random.gauss(pade_filter_param_a, pade_filter_param_a_dev)
    random_par_b = random.gauss(pade_filter_param_b, pade_filter_param_b_dev)
    random_par_c = random.gauss(pade_filter_param_c, pade_filter_param_c_dev)
    random_par_d = random.gauss(pade_filter_param_d, pade_filter_param_d_dev)
    random_par_e = random.gauss(pade_filter_param_e, pade_filter_param_e_dev)
    random_par_f = random.gauss(pade_filter_param_f, pade_filter_param_f_dev)

    """fits background to a form of (A+B*dx+C*dx^2+D*dx3)/(1+E*dx+F*dx2)"""
    deltas=np.linspace(-25000, 25000, 500)
    
    Q_param=40000
    scaled_deltas=Q_param*deltas/((0.5*(spectrum_start_freq+spectrum_stop_freq)))
 
     #### Here are random raw spectrum created
    random_y = mypade(scaled_deltas, random_par_a, random_par_b, random_par_c, random_par_d, random_par_e, random_par_f)

    #create a random number array
    randnums= np.random.randint(1,101,500)
    y_noise=0.00002 * randnums
    random_raw_y= random_y + y_noise
    #divide by 1.0e-07 to scale the data. Remove this hard coding later on
    div_num = 1.0e-07
    random_raw_y_final = div_num*random_raw_y

    ###Create the random raw spectrum
    random_raw_spectrum=PowerSpectrum(random_raw_y_final, spectrum_start_freq, spectrum_stop_freq)

    #Todo: Remove the hardcoded 500 to len(sub_spectrum)
    filter_shape_pade=np.ones(500) # to prevent errors later on

    curve_condition_runtime = 0 
    try:
        popt, pcov = curve_fit(mypade, scaled_deltas, random_raw_y_final)
        filter_shape_pade=mypade(scaled_deltas, *popt)
        curve_condition_runtime=1
    except RuntimeError:
        #print("line number", line_counter)
        #print("resonant frequency", 0.5*(spectrum_raw.xstart+spectrum_raw.xstop))
        curve_condition_runtime=2
        #print("fit did not converge")

    return random_raw_spectrum, filter_shape_pade, curve_condition_runtime


def shift_list(l, shift, empty=0):
    src_index = max(-shift, 0)
    dst_index = max(shift, 0)
    length = max(len(l) - abs(shift), 0)
    new_l = [empty] * len(l)
    new_l[dst_index:dst_index + length] = l[src_index:src_index + length]
    return new_l


# looks like this function takes scans +/- buffer around the original index. Then shifts them to the correct position and averages them?
# If were looking for spikes then do we actually need to average the bins? A quick signal would be only in one bin so the average would muddy the signal 
# Once 
def get_N_adjacent_spectra(start_time,stop_time,current_center_freq,current_start_freq,current_stop_freq,receiver_shape,output_file,timestamp_id,current_index,record):
    max_lines=60
    receiver_shape_mean=mean(receiver_shape.yvalues)    
    receiver_shape_yvals=receiver_shape.yvalues/receiver_shape_mean
    receiver_shape.yvalues=receiver_shape_yvals
#    short_query="SELECT A.timestamp, B.start_frequency_channel_one, B.stop_frequency_channel_one, B.frequency_resolution_channel_one, B.power_spectrum_channel_one, B.sampling_rate, B.integration_time, A.digitizer_log_reference, A.notes, B.lo_frequency_channel_one, A.mode_frequency_channel_one from axion_scan_log as A INNER JOIN digitizer_log as B ON A.digitizer_log_reference=B.digitizer_log_id WHERE A.timestamp < '"+str(stop_time)+"' AND A.timestamp>'"+str(start_time)+"' AND B.notes='taking data' ORDER BY A.timestamp desc LIMIT "+str(max_lines)
#    short_record=db.send_admxdb_query(short_query)
    num_lines=0
    shift=''
    buffer=3
    averaged_spectrum=np.zeros(len(record[0][4]))
    count_spectrum=np.zeros(len(record[0][4]))

    for line in range(-buffer,buffer):
        #start with scans behind the main scan and go to scans ahead of the main scan
        index=current_index+line
        if(index<0):
            continue
        if(index>=len(record)):
            continue

        spectrum_start_freq=record[index][1]+record[index][9] #line[1]+line[9]
        spectrum_stop_freq=record[index][2]+record[index][9]  #line[2]+line[9]
        center_freq=(spectrum_start_freq+spectrum_stop_freq)/2
        spectrum_raw_yvals=record[index][4] #line[4]
        delta_freq=current_center_freq-center_freq
        if(delta_freq>0):
            shift='right'
        elif(delta_freq<0):
            shift='left'
        else:
            shift='no'
        delta_freq=abs(delta_freq)
        binwidth=(abs(spectrum_start_freq-spectrum_stop_freq))/len(spectrum_raw_yvals)
        binshift=delta_freq/binwidth
        binshift=int(round(binshift))
        if(shift=='left'):
            binshift=-binshift
        if(shift=='no'):
            binshift=0
        spectrum_mean=mean(spectrum_raw_yvals)
        spectrum_raw_yvals=[i/spectrum_mean for i in spectrum_raw_yvals]
        shifted_raw_spectrum=shift_list(spectrum_raw_yvals,binshift)
        for i in range(len(shifted_raw_spectrum)):
            #weight=100
            weight=1            
            if(binshift==0):
                averaged_spectrum[i]=averaged_spectrum[i]+weight*shifted_raw_spectrum[i]
            else:
                averaged_spectrum[i]=averaged_spectrum[i]+shifted_raw_spectrum[i]
            if(shifted_raw_spectrum[i]!=0):
                if(binshift==0):
                    count_spectrum[i]=count_spectrum[i]+weight
                else:
                    count_spectrum[i]=count_spectrum[i]+1
        one_filter_spectrum=ADMXDataSeries(shifted_raw_spectrum,current_start_freq,current_stop_freq)
        fnamestring="/one_filter_spectrum"+str(num_lines)
        save_dataseries_to_hdf5(output_file,timestamp_id.decode('UTF-8')+fnamestring,one_filter_spectrum)
        num_lines=num_lines+1

    filter_shape_avg=[]
    averaged_power_spectrum=PowerSpectrum(averaged_spectrum,current_start_freq,current_stop_freq)
    if(len(averaged_spectrum)!=0):
        for i in range(len(averaged_spectrum)):
            averaged_spectrum[i]=averaged_spectrum[i]/count_spectrum[i]
        bg_fit_type="pade"
        averaged_spectrum=PowerSpectrum(averaged_spectrum,current_start_freq,current_stop_freq)

    return averaged_spectrum


#Do what needs to be done to each scan
first_pass=True
line_counter=0
pade_fit_params_array = np.zeros((6, len(records)))
for line in records:
    save_extra_info=False
    line_counter+=1
    timestamp = line[0]
    timestamp_val = line[0].isoformat()
    timestamp_id=np.string_(timestamp_val)
    timestamp_id = timestamp_id[:-7]

    if timestamp_id not in params["scanid"]:
        #print(timestamp_id)
        #print(params["scanid"])
        a=timestamp_id.decode('UTF-8')
        b=params["scanid"][0].decode('UTF-8')
        print("Error parameters don't exist for scan "+a)
        continue
    scanid_array.append(timestamp_id)
    ind=np.where(params["scanid"]==timestamp_id)[0][0]

# This loop is used to loop through ADMX database, will start with single data file

    f0=params["f0"][ind]
    coupling_factor=params["coupling"][ind]
    Tsys=params["Tsys"][ind]
    ff_v=params["ff_v"][ind]
    Q=params["Q"][ind]
    Q_refl=params["Q_refl"][ind]
    Q_trans=params["Q_trans"][ind]
    B=params["B"][ind]
    integration_time_seconds=line[6]
    
    #if its already cut, don't bother
    cut_reason=params["cut_reason"][ind].decode('UTF-8')
    if cut_reason=="":
        should_cut,cut_reason=test_timestamp_cuts(timestamp_cut_yaml,timestamp)
    if cut_reason=="":
        should_cut,cut_reason=frequency_cuts(frequency_cut_yaml, timestamp, f0)         

    if cut_reason=="" and integration_time_seconds=="":
       should_cut= True
       cut_reason="Integration time not logged"
    #Extract the raw spectrum
    spectrum_start_freq=line[1]+line[9]
    spectrum_stop_freq=line[2]+line[9]
 
    if check_simulate_raw_spectra==True:
        #Use simulated data
        #print("Using simulated data")
        spectrum_raw, filter_shape_pade,curve_condition_runtime  = get_simulated_spectrum(spectrum_start_freq, spectrum_stop_freq)
        spectrum_raw.metadata["scanid"]=timestamp_id
    else: 
        #Use real data
        #print("Using real data")
        spectrum_raw=PowerSpectrum(line[4],spectrum_start_freq,spectrum_stop_freq)

        spectrum_raw.yvalues[0]=spectrum_raw.yvalues[1]
        spectrum_raw.metadata["scanid"]=timestamp_id

    get_pade_filter_params(spectrum_raw,receiver_shape,line_counter-1)

    #Roughly mask noise peak before filter
    
    for masktimestr in  maskbins.keys():         
        maskstart =  datetime.datetime.strptime(masktimestr.split(" TO ")[0], '%Y-%m-%d %H:%M:%S %z')
        maskend =  datetime.datetime.strptime(masktimestr.split(" TO ")[1], '%Y-%m-%d %H:%M:%S %z')
        maskbin = maskbins[masktimestr]        
        if maskstart < timestamp and maskend > timestamp: 
            for ii,val in enumerate(line[4]): 
                masked=False
                avg=0
                for themaskbin in  maskbin:
                    if ii>=themaskbin[0] and ii<=themaskbin[1] : 
                        masked=True
                        avg=(line[4][themaskbin[0]]+line[4][themaskbin[1]])/2.
                if masked : 
                    line[4][ii] = avg
                else : 
                    line[4][ii] = val


    #create an empty array for an individual scan
    # line[4] seems to be data? 
    # line[9] likely cavity frequency 
    # line[1] likely a negative, freq width from cav frequency
    # line[2] positive, other freq width from cav frequency 
    # line[0] timestamp     
    # line[6] is integration time in s    
                                                                                                                                    
    spectrum_start_freq=line[1]+line[9]
    spectrum_stop_freq=line[2]+line[9]
    center_freq=(spectrum_start_freq+spectrum_stop_freq)/2
    empty_yvalues=np.zeros(len(line[4]))
    averaged_baseline_array=PowerSpectrum(empty_yvalues,spectrum_start_freq,spectrum_stop_freq)
    current_timestamp=line[0]
    start_time=current_timestamp-timedelta(minutes=6)
    stop_time=current_timestamp+timedelta(minutes=6)
    averaged_background=get_N_adjacent_spectra(start_time,stop_time,center_freq,spectrum_start_freq,spectrum_stop_freq,receiver_shape,output_file,timestamp_id,line_counter,records)
    spectrum_raw_mean=mean(spectrum_raw.yvalues)
    #temp comment out
    averaged_background=[ averaged_background.yvalues[i]*spectrum_raw_mean for i in range(len(averaged_background)) ]
    averaged_background_ps=PowerSpectrum(averaged_background,spectrum_start_freq,spectrum_stop_freq)

    # This is an alternative background removal option which requires the use of three different filters.
    if(bg_fit_type=='averaged_bg'):
        bg_fit_type='pade'
        avg_spectrum_filtered,filter_shape_avg,spectrum_filtered_except_sig,badness=convert_raw_spectrum_to_filtered_spectrum(raw_spectrum=averaged_background_ps,shape_receiver=receiver_shape,filter_predefined=averaged_background_ps,type_fit=bg_fit_type)
        if(badness==0):
            fnamestring="/avg_filter_spectrum"
            test='testing'
            mystring=fnamestring+test
            avg_filter_spec=ADMXDataSeries(filter_shape_avg,spectrum_start_freq,spectrum_stop_freq)

            bg_fit_type='predefined'
            spectrum_filtered,filter_shape,spectrum_filtered_except_sig,badness=convert_raw_spectrum_to_filtered_spectrum(raw_spectrum=spectrum_raw,shape_receiver=receiver_shape,filter_predefined=avg_filter_spec,type_fit=bg_fit_type)
            if(badness==0):
                filter_spectrum=ADMXDataSeries(filter_shape,spectrum_filtered.xstart,spectrum_filtered.xstop)
                save_dataseries_to_hdf5(output_file,timestamp_id.decode('UTF-8')+"/filter_spectrum",filter_spectrum)
                save_dataseries_to_hdf5(output_file,timestamp_id.decode('UTF-8')+"/avg_filter_spectrum",avg_filter_spec)


            bg_fit_type='pade'
            single_spectrum_filtered,single_filter_shape,blah,badness=convert_raw_spectrum_to_filtered_spectrum(raw_spectrum=spectrum_raw,shape_receiver=receiver_shape,filter_predefined=averaged_background_ps,type_fit=bg_fit_type)
            if(badness==0):
                single_filter_spectrum=ADMXDataSeries(single_filter_shape,spectrum_filtered.xstart,spectrum_filtered.xstop)
                save_dataseries_to_hdf5(output_file,timestamp_id.decode('UTF-8')+"/single_filter_spectrum",single_filter_spectrum)
    
                    
    # All other background removal options    
    spectrum_filtered,filter_shape,spectrum_filtered_except_sig,badness=convert_raw_spectrum_to_filtered_spectrum(raw_spectrum=spectrum_raw,shape_receiver=receiver_shape,type_fit=bg_fit_type)

    # stdev=spectrum_filtered_except_sig.yvalues.std()
    # mn=spectrum_filtered_except_sig.yvalues.mean()
    stdev=spectrum_filtered.yvalues.std()
    mn=spectrum_filtered.yvalues.mean()


    if not isinstance(filter_shape,np.ndarray):
        #error in filter shape
        if cut_reason=="":
            cut_reason="error in filter shape"
        filter_shape=np.ones(len(spectrum_filtered)) # to prevent errors later on


    #Here is where to inject synthetic signals
    if should_inject_synthetic(spectrum_filtered):
        noise_power=params["Tsys"][ind]*kboltz*(spectrum_raw.get_xspacing())
        power_ksvz= coupling_factor/(coupling_factor+1)*ksvz_axion_power(Q,f0,ff_v*B*B)
        #print("injecting {} W".format(power_ksvz))
        #print("expected instant noise power {} W".format(noise_power))
        expected_ksvz_snr=(power_ksvz/noise_power)
        #print("expected instant snr {}".format(expected_ksvz_snr))
        spectrum_raw=inject_synthetic(spectrum_raw,expected_ksvz_snr,filter_shape,f0,Q)
        spectrum_raw.metadata["scanid"]=timestamp_id
        #refilter 
        spectrum_filtered,filter_shape,spectrum_filtered_except_sig,badness=convert_raw_spectrum_to_filtered_spectrum(raw_spectrum=spectrum_raw,shape_receiver=receiver_shape,type_fit=bg_fit_type)
        # stdev=spectrum_filtered_except_sig.yvalues.std()
        # mn=spectrum_filtered_except_sig.yvalues.mean()
        stdev=spectrum_filtered.yvalues.std()
        mn=spectrum_filtered.yvalues.mean()
        save_extra_info=True
    if not isinstance(filter_shape,np.ndarray):
        #error in filter shape
        if cut_reason=="":
            cut_reason="error in filter shape"
        filter_shape=np.ones(len(spectrum_filtered)) # to prevent errors later on



    #---TODO move the following into the filtering function---
    #characterize standard deviation, scale to unity, change to tracking errors with 'measurement' instead of 'spectrum'
       #you can predict the stdev/mn from the integration time and bandwidth
   
    #if integration_time_seconds is None:
        #print("integration time is none, assuming 100")
        #integration_time_seconds = 100

    expected_stdev=1.0/math.sqrt(integration_time_seconds*spectrum_filtered.get_xspacing())
    #print(expected_stdev)
    #record that for later, may want to cut on it
    stdev_increase_array.append((stdev/mn)/expected_stdev)
    measurement_filtered=PowerMeasurement(spectrum_filtered.yvalues/mn-np.ones(len(spectrum_filtered.yvalues)),np.ones(len(spectrum_filtered.yvalues))*stdev/mn,spectrum_filtered.xstart,spectrum_filtered.xstop,metadata=spectrum_filtered.metadata)
    measurement_filtered.yunits="Excess Power SNR"
    measurement_filtered.metadata["timestamp"]=timestamp_id
    measurement_filtered.metadata["coupling_factor"]=coupling_factor
    measurement_filtered.metadata["Tsys"]=params["Tsys"][ind]
    measurement_filtered.metadata["integration_time"]=integration_time_seconds
    measurement_filtered.metadata["f0"]=f0
    if Q_mode=='reflection_q':
        measurement_filtered.metadata["Q"]=Q_refl
        measurement_filtered.metadata["dfsz_power"]=dfsz_axion_power(Q_refl,f0,ff_v*B*B)*1e24
    elif Q_mode=='transmission_q':
        measurement_filtered.metadata["Q"]=Q_trans
        measurement_filtered.metadata["dfsz_power"]=dfsz_axion_power(Q_trans,f0,ff_v*B*B)*1e24
    else:
        measurement_filtered.metadata["Q"]=Q
        measurement_filtered.metadata["dfsz_power"]=dfsz_axion_power(Q,f0,ff_v*B*B)*1e24
    #----------

    # Mask noise peak
    for masktimestr in  maskbins.keys():         
        maskstart =  datetime.datetime.strptime(masktimestr.split(" TO ")[0], '%Y-%m-%d %H:%M:%S %z')
        maskend =  datetime.datetime.strptime(masktimestr.split(" TO ")[1], '%Y-%m-%d %H:%M:%S %z')
        maskbin = maskbins[masktimestr]
        if maskstart < timestamp and maskend > timestamp: 
            for ii,val in enumerate(measurement_filtered.yvalues): 
                masked=False
                for themaskbin in  maskbin:
                    if ii>=themaskbin[0] and ii<=themaskbin[1] : masked=True
                if masked : 
                    measurement_filtered.yvalues[ii] = 0
                    measurement_filtered.yuncertainties[ii] = measurement_filtered.yuncertainties[ii]*1e5 ## No difference between 1e3 and 1e5
                else : 
                    measurement_filtered.yvalues[ii] = val

            
       
   
    #Apply relevant cuts
    if (stdev/mn)/expected_stdev>stdev_max_cut:
        if cut_reason=="":
            cut_reason="stdev increase too high"
        print("stdev too high", (stdev/mn)/expected_stdev)

        #and record the distribution for data quality (if not cut)

    if cut_reason=="":
        histo_toadd=np.bincount(np.digitize(measurement_filtered.yvalues/measurement_filtered.yuncertainties,snr_bins),minlength=len(snr_bins)+1)
        snr_scatter_histogram=snr_scatter_histogram+histo_toadd


    #scale to unit experiment and noise
    binwidth_hz=measurement_filtered.get_xspacing()
    norm=(kboltz*binwidth_hz)

    if Q_mode=='reflection_q':
        lorentzian_array=np.array([ norm/(coupling_factor/(1+coupling_factor)*lorentz(f,f0,Q_refl)) for f in measurement_filtered.get_xvalues() ])
    elif Q_mode=='transmission_q':
        lorentzian_array=np.array([ norm/(coupling_factor/(1+coupling_factor)*lorentz(f,f0,Q_trans)) for f in measurement_filtered.get_xvalues() ])
    else:
        lorentzian_array=np.array([ norm/(coupling_factor/(1+coupling_factor)*lorentz(f,f0,Q)) for f in measurement_filtered.get_xvalues() ])
        

    measurement_lor_scaled=measurement_filtered*lorentzian_array #You multiply this by the noise temperature in kelvin and get Watts per bin of excess power, were the cavity tunide to that frequency
    measurement_lor_scaled.yunits="Excess Power in Cavity Watts / Kelvin"


    #Record debug info for first cycle
    if first_pass:
        first_pass=False

    #save to file
    target_name=timestamp_id.decode('UTF-8')+"/measurement_scaled"
    filtered_name=timestamp_id.decode('UTF-8')+"/measurement_filtered"
    raw_name=timestamp_id.decode('UTF-8')+"/raw_spectrum"

    #if line_counter%200==0 or save_extra_info==True:
    #print("#----#")
    save_measurement_to_hdf5(output_file,target_name,measurement_lor_scaled) #this has units of Watts per kelvin
    save_measurement_to_hdf5(output_file,filtered_name,measurement_filtered)
    save_dataseries_to_hdf5(output_file,raw_name,spectrum_raw)
    filter_spectrum=ADMXDataSeries(filter_shape,spectrum_filtered.xstart,spectrum_filtered.xstop)
    save_dataseries_to_hdf5(output_file,timestamp_id.decode('UTF-8')+"/filter_spectrum",filter_spectrum)
    #print("saving extra info on line ",line_counter," that is ",line[0])
    if cut_reason!= "":
        print("this one was cut because {}".format(cut_reason))
        maxpower=np.max(measurement_filtered.yvalues)
        #print("max snr {} ".format(maxpower))
        maxpower=np.max(measurement_lor_scaled.yvalues)
        #print("max power from axions is {} Watts".format(maxpower*params["Tsys"][ind]))
        #print("#----#")



    cut_reason_array.append(cut_reason)
    if cut_reason in cut_reason_histogram:
        cut_reason_histogram[cut_reason]+=1
    else:
        cut_reason_histogram[cut_reason]=1

"""
print("params array", pade_fit_params_array)
print("params avg [0]", np.mean(pade_fit_params_array[0]))
print("params avg [1]", np.mean(pade_fit_params_array[1]))
print("params avg [2]", np.mean(pade_fit_params_array[2]))
print("params avg [3]", np.mean(pade_fit_params_array[3]))
print("params avg [4]", np.mean(pade_fit_params_array[4]))
print("params avg [5]", np.mean(pade_fit_params_array[5]))

print("params std [0]", np.std(pade_fit_params_array[0]))
print("params std [1]", np.std(pade_fit_params_array[1]))
print("params std [2]", np.std(pade_fit_params_array[2]))
print("params std [3]", np.std(pade_fit_params_array[3]))
print("params std [4]", np.std(pade_fit_params_array[4]))
print("params std [5]", np.std(pade_fit_params_array[5]))
"""

if len(cut_reason_array)!=len(scanid_array):
    print("error cuts and scanids are misaligned")
stringtype = h5py.special_dtype(vlen=bytes)
dset_scanids=output_file.create_dataset("scanid",(len(scanid_array),),dtype="S25")
dset_scanids[...]=scanid_array
dset_cutreason=output_file.create_dataset("cut_reason",(len(cut_reason_array),),dtype=stringtype)
dset_cutreason[...]=cut_reason_array

output_file.create_dataset("stdev_increase",(len(stdev_increase_array),),data=stdev_increase_array)
save_dataseries_to_hdf5(output_file,"snr_deviation_histogram",ADMXDataSeries(snr_scatter_histogram,snr_bins[0],snr_bins[-1]))
output_file.close()

#Save the pade filter parameters distribution
outfile_name=pade_params_output_file_name
outfile=h5py.File(outfile_name,"w")

outfile.create_dataset("pade_fit_params_array0",data=list(pade_fit_params_array[0]))
outfile.create_dataset("pade_fit_params_array1",data=list(pade_fit_params_array[1]))
outfile.create_dataset("pade_fit_params_array2",data=list(pade_fit_params_array[2]))
outfile.create_dataset("pade_fit_params_array3",data=list(pade_fit_params_array[3]))
outfile.create_dataset("pade_fit_params_array4",data=list(pade_fit_params_array[4]))
outfile.create_dataset("pade_fit_params_array5",data=list(pade_fit_params_array[5]))
outfile.close()

print("Cut Report:")
for key in cut_reason_histogram:
    if key != "":
        print(key,": {}".format(cut_reason_histogram[key]))
    else:
        print(cut_reason_histogram[key]," passed")
