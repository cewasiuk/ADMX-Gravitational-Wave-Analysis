import os
from statistics import mean
import sys
import math
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
from datetime import datetime, timedelta, timezone

from collections import namedtuple
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
import h5py
from dateutil import parser
import yaml
import argparse
from scipy.constants import codata
from scipy.optimize import curve_fit
from axion_parameters import ksvz_axion_power,dfsz_axion_power
import random
import pandas as pd
import struct
import json
import pyfftw
import array
from matplotlib import pyplot as plt
import re
# all cut reasons excluded?? Not included in any param file I have
#default run config file
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

# print(run_definition["nibbles"][target_nibble])

start_time=run_definition["nibbles"][target_nibble]["start_time"]
stop_time=run_definition["nibbles"][target_nibble]["stop_time"]
bg_fit_type=run_definition["nibbles"][target_nibble]["bg_fit_type"]

#parameter_file=run_definition["nibbles"][target_nibble]["parameter_file_name"]
parameter_file=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"2018_05_19.txt")
#receiver_shape_file=run_definition["nibbles"][target_nibble]["receiver_shape_file"]
receiver_shape_file=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"receiver_shape.h5")
#output_file_name=run_definition["nibbles"][target_nibble]["spectrum_file_name"]
output_file_name=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"spectra.h5")
timestamp_cut_yaml=run_definition["timestamp_cuts"]
stdev_max_cut=run_definition["parameter_cuts"]["max_stdev_increase"]
frequency_cut_yaml=run_definition["frequency_cuts"]
# maskbins=run_definition["nibbles"][target_nibble]["maskbins"]
pade_params_output_file_name=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"pade_params.h5")


useful_parameters=["Start_Frequency", "Stop_Frequency", "Digitizer_Log_ID","Integration_Time","Filename_Tag","Quality_Factor","Cavity_Resonant_Frequency","JPA_SNR","Thfet","Attenuation","Reflection","Transmission?"]
df = pd.read_csv(parameter_file, delimiter = '\t', names=useful_parameters, header=None)
prefix = "admx"
df["Filename_Tag"] = df["Filename_Tag"].apply(lambda x: prefix + x)





####### Skip Synthetic Signal Injection ##########

def lorentz(x,x0,Q):
    return 1/(4*(x-x0)*(x-x0)*Q*Q/(x0*x0)+1)


#load the background shape
f=h5py.File(receiver_shape_file,"r")
receiver_shape=load_dataseries_from_hdf5(f,"receiver_shape")
f.close()

#load the data
HR_folder = "hr_data"
data_folder = "2018_05_19"
binned_data_folder = 'binned_hr_data'
folder_path = HR_folder + "/" + binned_data_folder 
# Files can be loaded from this path in the form of Power vs time, will need to FT


# Turner Lineshape variables below
# Unix timestamp for June 2 at midnight, 2017

########################### DONT EXIST IN RUN1b_def.yaml?    ########################### 

# midnight_june_2 = 1496361600

# v_halo=run_definition["lineshape_parameters"]["v_halo"]
# v_sun=run_definition["lineshape_parameters"]["v_sun"]
# v_earth=run_definition["lineshape_parameters"]["v_earth"]
# v_rot=run_definition["lineshape_parameters"]["v_rot"]
# parallel=run_definition["lineshape_parameters"]["parallel"]
# alpha=run_definition["lineshape_parameters"]["alpha"]
# psi=run_definition["lineshape_parameters"]["psi"]
# omega_o=run_definition["lineshape_parameters"]["omega_o"]
# omega_r=run_definition["lineshape_parameters"]["omega_r"]
# turner_parameter_names=["v_halo","v_sun","v_earth","v_rot","parallel","alpha","psi","omega_o","omega_r"]
# turner_parameter_values=[v_halo,v_sun,v_earth,v_rot,parallel,alpha,psi,omega_o,omega_r]
# turner_parameters=dict(zip(turner_parameter_names,turner_parameter_values))

# #pade filter parameters for simulating data
# pade_filter_param_a=run_definition["software_synthetics"]["pade_filter_params"]["a"]
# pade_filter_param_b=run_definition["software_synthetics"]["pade_filter_params"]["b"]
# pade_filter_param_c=run_definition["software_synthetics"]["pade_filter_params"]["c"]
# pade_filter_param_d=run_definition["software_synthetics"]["pade_filter_params"]["d"]
# pade_filter_param_e=run_definition["software_synthetics"]["pade_filter_params"]["e"]
# pade_filter_param_f=run_definition["software_synthetics"]["pade_filter_params"]["f"]
# pade_filter_param_a_dev=run_definition["software_synthetics"]["pade_filter_params_dev"]["a_dev"]
# pade_filter_param_b_dev=run_definition["software_synthetics"]["pade_filter_params_dev"]["b_dev"]
# pade_filter_param_c_dev=run_definition["software_synthetics"]["pade_filter_params_dev"]["c_dev"]
# pade_filter_param_d_dev=run_definition["software_synthetics"]["pade_filter_params_dev"]["d_dev"]
# pade_filter_param_e_dev=run_definition["software_synthetics"]["pade_filter_params_dev"]["e_dev"]
# pade_filter_param_f_dev=run_definition["software_synthetics"]["pade_filter_params_dev"]["f_dev"]



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


def normalize_kernel(kernel):
    mysum=0
    for x in kernel:
        mysum+=x
    return kernel/mysum


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


######## Put in parameters by hand for now ##########
fname="./admx"+df['Filename_Tag']
line_counter = 0 
file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
pade_fit_params_array = np.zeros((6, file_count))
pattern = r"_(\d{4}_\d{2}_\d{2})_(\d{2}_\d{2}_\d{2})_"
first_pass=True
# print("\n Have {} files \n".format(file_count))
for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    file_name = file_name.replace('admx','').replace('_binned.h5','')

    if os.path.isfile(file_path):
        print(file_name)
        if df["Filename_Tag"].str.contains(file_name).any() == True:
            matching_indices = df[df["Filename_Tag"].str.contains(file_name, na=False)].index
            match = re.search(pattern,  df["Filename_Tag"][matching_indices[0]])


            # Extract timestamp from datafile
            if match:
                date = str(match.group(1))
                time = match.group(2) # Extract the time
                utc_offset_hours = -8   
                date_obj = datetime.strptime(date, "%Y_%m_%d")
                time_obj = datetime.strptime(time, "%H_%M_%S")

                combined_datetime = date_obj.replace(hour=time_obj.hour, minute=time_obj.minute, second=time_obj.second)
                utc_offset = timezone(timedelta(hours=utc_offset_hours))
                final_datetime = combined_datetime.replace(tzinfo=utc_offset)
                current_timestamp = final_datetime.isoformat()
                current_timestamp =  current_timestamp.replace("T"," ")
                timestamp_id = str(current_timestamp[:-6])

            else:
                print("No match found!")
                continue
            # Exit current file 
            spectrum_start_freq = df.loc[df["Filename_Tag"].str.contains(file_name), "Start_Frequency"].iloc[0]
            spectrum_stop_freq = df.loc[df["Filename_Tag"].str.contains(file_name), "Stop_Frequency"].iloc[0]
            coupling_factor = 0.1
            jpa_snri = df.loc[df["Filename_Tag"].str.contains(file_name), "JPA_SNR"].iloc[0]
            atten = df.loc[df["Filename_Tag"].str.contains(file_name), "Attenuation"].iloc[0]
            thfet = df.loc[df["Filename_Tag"].str.contains(file_name), "Thfet"].iloc[0]
            Tsys =thfet/(10**(jpa_snri/10))*1/(10**(atten/10))
            Q_refl = df.loc[df["Filename_Tag"].str.contains(file_name), "Reflection"].iloc[0] 
            Q_trans = df.loc[df["Filename_Tag"].str.contains(file_name), "Transmission?"].iloc[0] 
            f0 =  df.loc[df["Filename_Tag"].str.contains(file_name), "Cavity_Resonant_Frequency"].iloc[0] 
            center_freq = (spectrum_start_freq + spectrum_stop_freq)/2
            integration_time_seconds = df.loc[df["Filename_Tag"].str.contains(file_name), "Integration_Time"].iloc[0] 
            #Dont currently know what this is; will figure out later
            B = get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"Bfield") #assuming magnetic field
            Q = 1e5 # shouldnt have to manually put this in
            ff_v = 1 
        else:
            #If data doesnt have any run info w/i definitions file, we skip it
            print('what')
            continue

    #Extract the raw spectrum
    # Need to open original files from desktop

    # Will need to add timesteps to everything? can extract from definitions of run file using regex. 
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
        step = h1_json['x_spacing']*1e-6

        #prepare to read the data. First, get the number of data points
        (data_size,) = struct.unpack('Q',file.read(8))
        
        #now, read the data.
        data = array.array('f') #declare an array of floats
        data.frombytes(file.read(data_size*4)) #each float is 4 bytes, so read number of data points x 4 bytes
        print(data)
        end_read = file.tell()
    ## Once we have data, begin spectral preparation 
    spectrum_ts = ADMXDataSeries(yvalues = data, xstart = 0, xstop = len(data)*step ) # Y-units in Voltage, x-units in seconds 
    fft = pyfftw.interfaces.numpy_fft.rfft(data)
    spectrum_fs = ADMXDataSeries( yvalues = fft, xstart = spectrum_start_freq, xstop = spectrum_stop_freq)
    pow_freq= ADMXDataSeries(yvalues = np.square(np.abs(spectrum_fs.yvalues)), xstart = spectrum_start_freq, xstop = spectrum_stop_freq)
    spectrum_raw=PowerSpectrum( power = spectrum_fs.yvalues,start_freq = spectrum_start_freq, stop_freq = spectrum_stop_freq) 
    get_pade_filter_params(spectrum_raw,receiver_shape,line_counter-1)
    bg_fit_type = "pade"
    
    # All other background removal options    
    spectrum_filtered,filter_shape,spectrum_filtered_except_sig,badness=convert_raw_spectrum_to_filtered_spectrum(raw_spectrum=spectrum_raw,shape_receiver=receiver_shape,type_fit=bg_fit_type)

    stdev=spectrum_filtered.yvalues.std()
    mn=spectrum_filtered.yvalues.mean()


    expected_stdev=1.0/math.sqrt(integration_time_seconds*spectrum_filtered.get_xspacing())
    #print(expected_stdev)
    #record that for later, may want to cut on it
    log_id = df["Digitizer_Log_ID"][matching_indices]

    stdev_increase_array.append((stdev/mn)/expected_stdev)
    measurement_filtered=PowerMeasurement(spectrum_filtered.yvalues/mn-np.ones(len(spectrum_filtered.yvalues)),np.ones(len(spectrum_filtered.yvalues))*stdev/mn,spectrum_filtered.xstart,spectrum_filtered.xstop,metadata=spectrum_filtered.metadata)
    measurement_filtered.yunits="Excess Power SNR"
    measurement_filtered.metadata["timestamp"]=timestamp_id
    measurement_filtered.metadata["coupling_factor"]=coupling_factor
    measurement_filtered.metadata["Tsys"]=Tsys
    measurement_filtered.metadata["integration_time"]=integration_time_seconds
    measurement_filtered.metadata["f0"]=f0 # Have no idea what f0 is 
    measurement_filtered.metadata["Q"]= Q

# Computed power of Dine-Fischler-Srednicki-Zhitnisky axions>
    # measurement_filtered.metadata["dfsz_power"]=dfsz_axion_power(Q,f0,ff_v*B*B)*1e24

        # Mask noise peak - do this eventually?
    # for masktimestr in  maskbins.keys():         
    #     maskstart =  datetime.datetime.strptime(masktimestr.split(" TO ")[0], '%Y-%m-%d %H:%M:%S %z')
    #     maskend =  datetime.datetime.strptime(masktimestr.split(" TO ")[1], '%Y-%m-%d %H:%M:%S %z')
    #     maskbin = maskbins[masktimestr]
    #     if maskstart < timestamp and maskend > timestamp: 
    #         for ii,val in enumerate(measurement_filtered.yvalues): 
    #             masked=False
    #             for themaskbin in  maskbin:
    #                 if ii>=themaskbin[0] and ii<=themaskbin[1] : masked=True
    #             if masked : 
    #                 measurement_filtered.yvalues[ii] = 0
    #                 measurement_filtered.yuncertainties[ii] = measurement_filtered.yuncertainties[ii]*1e5 ## No difference between 1e3 and 1e5
    #             else : 
    #                 measurement_filtered.yvalues[ii] = val
    #   Can also include cuts here

    #scale to unit experiment and noise

# Will need to likely rebin data to specified bin width 
    kboltz=codata.value('Boltzmann constant')
    binwidth_hz=measurement_filtered.get_xspacing()
    print(binwidth_hz)
    norm=(kboltz*binwidth_hz)

    # No Q_mode in run1B ? I believe its in transmission mode?
    # if Q_mode=='reflection_q':
    #     lorentzian_array=np.array([ norm/(coupling_factor/(1+coupling_factor)*lorentz(f,f0,Q_refl)) for f in measurement_filtered.get_xvalues() ])
    # elif Q_mode=='transmission_q':
    lorentzian_array=np.array([ norm/(coupling_factor/(1+coupling_factor)*lorentz(f,f0,Q)) for f in measurement_filtered.get_xvalues() ])
    # else:
    #     lorentzian_array=np.array([ norm/(coupling_factor/(1+coupling_factor)*lorentz(f,f0,Q)) for f in measurement_filtered.get_xvalues() ])
        

    measurement_lor_scaled=measurement_filtered*lorentzian_array #You multiply this by the noise temperature in kelvin and get Watts per bin of excess power, were the cavity tunide to that frequency
    measurement_lor_scaled.yunits="Excess Power in Cavity Watts / Kelvin"

    if first_pass:
        first_pass=False

    #save to file
    target_name= timestamp_id +"/measurement_scaled"
    filtered_name=timestamp_id+"/measurement_filtered"
    raw_name=timestamp_id+"/raw_spectrum"

    save_measurement_to_hdf5(output_file,target_name,measurement_lor_scaled) #this has units of Watts per kelvin
    save_measurement_to_hdf5(output_file,filtered_name,measurement_filtered)
    save_dataseries_to_hdf5(output_file,raw_name,spectrum_raw)
    filter_spectrum=ADMXDataSeries(filter_shape,spectrum_filtered.xstart,spectrum_filtered.xstop)
    save_dataseries_to_hdf5(output_file,timestamp_id.decode('UTF-8')+"/filter_spectrum",filter_spectrum)
    print("saving extra info on line ",line_counter," that is ",line[0])
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

    break

    # This only loops through one of the high res files, cant loop through multiple due to the fact that the file is never opened within the loop
    # can fix but dont know why it isnt implemented in the first place


