def process_and_store_spectra(run_definition_path="run1b_definitions.yaml", target_nibble="nibble5", receiver_path = "receiver_template", output_folder = "Processed_Data"):
    import os
    from statistics import mean
    import sys
    import math
    import grand_spectrum_core.admx_db_interface
    from grand_spectrum_core.admx_db_datatypes import PowerSpectrum,PowerMeasurement,ADMXDataSeries
    from grand_spectrum_core.admx_datatype_hdf5 import save_dataseries_to_hdf5,load_dataseries_from_hdf5,save_measurement_to_hdf5,load_measurement_from_hdf5 
    from grand_spectrum_core.sg_background_fit import  filter_bg_with_sg,filter_bg_with_sg_keep_extrema 
    from grand_spectrum_core.pade_background_fit import filter_bg_with_pade,filter_bg_with_pade_plus_signal,filter_bg_with_predefined_filter 
    from grand_spectrum_core.pade_background_fit import mypade
    from grand_spectrum_core.boosted_maxwell_boltzmann import maxwell_boltzman_shape 
    from grand_spectrum_core.turner_axion import turner_axion_shape
    from grand_spectrum_core.config_file_handling import test_timestamp_cuts,frequency_cuts 
    from grand_spectrum_core.config_file_handling import get_intermediate_data_file_name
    from datetime import datetime, timedelta, timezone
    from grand_spectrum_core.extract_receiver_shape import extract_receiver
    from collections import namedtuple
    import numpy as np
    import pycbc
    from pycbc.psd import interpolate, inverse_spectrum_truncation
    np.set_printoptions(threshold=sys.maxsize)
    import h5py
    from dateutil import parser
    import yaml
    import argparse
    from scipy.constants import codata
    from scipy.optimize import curve_fit
    from grand_spectrum_core.axion_parameters import ksvz_axion_power,dfsz_axion_power
    import random
    import pandas as pd
    import struct
    import json
    import pyfftw
    import array
    from matplotlib import pyplot as plt
    import re

    argparser=argparse.ArgumentParser()
    argparser.add_argument("-r","--run_definition",help="run definition yaml file",default=run_definition_path)
    argparser.add_argument("-n","--nibble_name",help="name of nibble to run",default=target_nibble)
    argparser.add_argument("-u","--update",help="update this file (not yet implemented)",action="store_true")
    argparser.add_argument("-s","--synthetic",help="synthetic signal file")
    args=argparser.parse_args()
    # extract_receiver()
    #pull in run configuration
    run_definition_file=open(args.run_definition,"r")
    run_definition=yaml.load(run_definition_file, Loader = yaml.Loader)
    run_definition_file.close()
    target_nibble=args.nibble_name

    # print(run_definition["nibbles"][target_nibble])

    start_time=run_definition["nibbles"][target_nibble]["start_time"]
    stop_time=run_definition["nibbles"][target_nibble]["stop_time"]
    bg_fit_type=run_definition["nibbles"][target_nibble]["bg_fit_type"]
    B=run_definition["nibbles"][target_nibble]["Bfield"] # B Field


    #parameter_file=run_definition["nibbles"][target_nibble]["parameter_file_name"]
    parameter_file=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"2018_05_19.txt")
    #receiver_shape_file=run_definition["nibbles"][target_nibble]["receiver_shape_file"]
    receiver_shape_file = get_intermediate_data_file_name(run_definition["nibbles"][target_nibble], "receiver_shape.h5")
    receiver_shape_file = os.path.join("hr_data",receiver_path, os.path.basename(receiver_shape_file))
    #output_file_name=run_definition["nibbles"][target_nibble]["spectrum_file_name"]
    output_file_name=get_intermediate_data_file_name(run_definition["nibbles"][target_nibble],"spectra.h5")
    output_file_name = os.path.join(output_folder, os.path.basename(output_file_name))
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

        # have to rename file so it matches form in the description file to extract relevant parameters 
        match_name = file_name.replace('_binned.h5','').replace('admx','')

        if os.path.isfile(file_path):
            if df["Filename_Tag"].str.contains(match_name).any() == True:
                matching_indices = df[df["Filename_Tag"].str.contains(match_name, na=False)].index
                match = re.search(pattern,  df["Filename_Tag"][matching_indices[0]])
                # print(match_name)

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

                spectrum_start_freq = df.loc[df["Filename_Tag"].str.contains(match_name), "Start_Frequency"].iloc[0] *1e6
                spectrum_stop_freq = df.loc[df["Filename_Tag"].str.contains(match_name), "Stop_Frequency"].iloc[0] *1e6
                coupling_factor = 0.1
                jpa_snri = df.loc[df["Filename_Tag"].str.contains(match_name), "JPA_SNR"].iloc[0]
                atten = df.loc[df["Filename_Tag"].str.contains(match_name), "Attenuation"].iloc[0]
                thfet = df.loc[df["Filename_Tag"].str.contains(match_name), "Thfet"].iloc[0]
                Tsys =thfet/(10**(jpa_snri/10))*1/(10**(atten/10))
                # Q_refl = df.loc[df["Filename_Tag"].str.contains(match_name), "Reflection"].iloc[0] 
                # Q_trans = df.loc[df["Filename_Tag"].str.contains(match_name), "Transmission?"].iloc[0] 
                f0 =  df.loc[df["Filename_Tag"].str.contains(match_name), "Cavity_Resonant_Frequency"].iloc[0] *1e6
                print(f0)
            
                center_freq = (spectrum_start_freq + spectrum_stop_freq)/2
                integration_time_seconds = df.loc[df["Filename_Tag"].str.contains(match_name), "Integration_Time"].iloc[0] 
                #Dont currently know what this is; will figure out later
                Q = df.loc[df["Filename_Tag"].str.contains(match_name), "Quality_Factor"].iloc[0] 
                V = 136 #Liters

            else:
                break
                #If data doesnt have any run info w/i definitions file, we skip it


        with h5py.File(file_path, "r") as file:
            # List all groups (datasets and groups) in the file
            group_name = str(list(file.keys())[0])
            print(group_name)
            if group_name in file:
                yvalues = file[group_name]

            attrs = dict(file[group_name].attrs)
            fstart = attrs['fstart']
            fstop = attrs['fstop'] 
            print('fstop',fstop)
            #Freqstep of Binned Data
            step = (fstop - fstart)/len(yvalues)
            # Fredstep before binning, used to compute timestep
            true_fstep = 1/integration_time_seconds
            yvalues = list(yvalues)

        
            
        ## Once we have data, begin spectral preparation 

        pow_freq= ADMXDataSeries(yvalues = yvalues, xstart = spectrum_start_freq+ fstart, xstop = spectrum_start_freq+ fstop, xunits = 'Hz',yunits = 'Power')
        spectrum_raw=PowerSpectrum( power = pow_freq.yvalues,start_freq = spectrum_start_freq+ fstart, stop_freq = spectrum_start_freq+ fstop) 
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



    # Will need to likely rebin data to specified bin width 
        kboltz=codata.value('Boltzmann constant')
        binwidth_hz=measurement_filtered.get_xspacing()
        norm=(kboltz*binwidth_hz)
        
        lorentzian_array=np.array([ norm/(coupling_factor/(1+coupling_factor)*lorentz(f,f0,Q)) for f in measurement_filtered.get_xvalues() ])
        # else:
        #     lorentzian_array=np.array([ norm/(coupling_factor/(1+coupling_factor)*lorentz(f,f0,Q)) for f in measurement_filtered.get_xvalues() ])

        measurement_lor_scaled=measurement_filtered*lorentzian_array*Tsys      #You multiply this by the noise temperature in kelvin and get Watts per bin of excess power, were the cavity tunide to that frequency
        measurement_lor_scaled.yunits="Excess Power in Cavity Watts / Kelvin"

        # Now need to scale each bin/sample by (1/Q)*(1/B^2)*(1/V^(5/3)) since these all vary with each dataset, enables apples to apples comparison between all datasets

        lor_scaled = measurement_lor_scaled*(1/Q)*(1/B**2)*(1/V)
        
        
    ########################################################################
        # PSD is Welch Estimate from the lorenzian scaled measurement converted into timeseries data (i.e. after processing and removing filter shape + scaling by power )
    ########################################################################


        t_values = pyfftw.interfaces.numpy_fft.irfft(measurement_lor_scaled.yvalues)
        # time_step = 1/(2*(len(measurement_lor_scaled.yvalues) -1)*step)

        time_step = 1/((len(measurement_lor_scaled)+1)*2*step)
        t_values =  pycbc.types.TimeSeries(t_values, time_step ) 
        psd =  pycbc.psd.estimate.welch(t_values, seg_len=4096, seg_stride=2048, window='hann', avg_method='median', num_segments=None, require_exact_data_fit=False)




        # t_values = t_values.crop(2,1)
        # plt.plot(t_values.sample_times,t_values)
        # plt.show()
        freq_ax = np.linspace(fstart,fstop,len(measurement_lor_scaled.yvalues))
        seg_length = t_values.duration/5
        psd_values = t_values.psd(seg_length)

        psd_binned  = interpolate(psd, step, len(measurement_lor_scaled.yvalues))
        # plt.plot(psd_binned)
        # plt.show()
        # psd_binned = inverse_spectrum_truncation(psd_binned, int(4 * step),
        #                               low_frequency_cutoff=15)




        if first_pass:
            first_pass=False

        #save to file
        target_name= timestamp_id +"/measurement_scaled"
        filtered_name=timestamp_id+"/measurement_filtered"
        raw_name=timestamp_id+"/raw_spectrum"

        with h5py.File(output_file_name, "a") as output_file:

            match_group = output_file.create_group(match_name)

            # Saving Original Time Series Data
            print(f"Saving Original Time Series for: {match_name}")
            dataset_time_series = match_group.create_dataset("Time_Series", data=t_values)
            dataset_time_series.attrs["tstart"] = 0
            dataset_time_series.attrs["tstop"] = integration_time_seconds
            dataset_time_series.attrs["time_step"] = integration_time_seconds/len(t_values)
            dataset_time_series.attrs["xunits"] = "seconds"
            dataset_time_series.attrs["yunits"] = "Watts per Kelvin"

            # Saving Filtered Data
            print(f"Saving Filtered Measurement Data for: {match_name}\n")
            dataset_filtered = match_group.create_dataset("Filtered_Data", data=measurement_filtered.yvalues)
            dataset_filtered.attrs["fstart"] = fstart + spectrum_start_freq
            dataset_filtered.attrs["fstop"] = fstop + spectrum_start_freq
            dataset_filtered.attrs["xunits"] = "Hz"
            dataset_filtered.attrs["yunits"] = "Filtered Signal"

            print(f"Saving Lorentzian Scaled Data for: {match_name}\n")
            dataset_filtered = match_group.create_dataset("Lor_Scaled_Data", data=lor_scaled.yvalues)
            dataset_filtered.attrs["fstart"] = fstart + spectrum_start_freq
            dataset_filtered.attrs["fstop"] = fstop + spectrum_start_freq
            dataset_filtered.attrs["xunits"] = "Hz"
            dataset_filtered.attrs["yunits"] = "Filtered Signal"


            # Saving Raw Spectrum Data
            print(f"Saving Raw Spectrum Data for: {match_name}\n")
            save_dataseries_to_hdf5(match_group, "raw_spectrum", spectrum_raw)
            
            # Saving Filtered Spectrum Data
            print(f"Saving Filtered Spectrum Data for: {match_name}\n")
            filter_spectrum = ADMXDataSeries(filter_shape, spectrum_filtered.xstart, spectrum_filtered.xstop)
            save_dataseries_to_hdf5(match_group, "filter_spectrum", filter_spectrum)

            # Saving Standard Deviation Increase Array
            print(f"Saving Standard Deviation Increase for: {match_name}\n")
            match_group.create_dataset("stdev_increase", (len(stdev_increase_array),), data=stdev_increase_array)

            # Saving Signal to Noise Ratio Deviation Histogram
            print(f"Saving SNR Deviation Histogram for: {match_name}\n")
            save_dataseries_to_hdf5(match_group, "snr_deviation_histogram", ADMXDataSeries(snr_scatter_histogram, snr_bins[0], snr_bins[-1]))

            # Saving Metadata Attributes
            print(f"Saving Metadata Attributes for: {match_name}\n")
            match_group.attrs["fstart"] = fstart + spectrum_start_freq
            match_group.attrs["fstop"] = fstop + spectrum_start_freq
            match_group.attrs["std_dev"] = stdev
            match_group.attrs["fsample"] = step
            match_group.attrs["int_time"] = integration_time_seconds

            # Saving PSD Binned Data
            print(f"Saving PSD Binned Data for: {match_name} \n")
            dataset_psd = match_group.create_dataset("PSD_Binned", data=psd_binned)
            dataset_psd.attrs["fstart"] = fstart
            dataset_psd.attrs["fstop"] = fstop
            dataset_psd.attrs["xunits"] = "Hz"
            dataset_psd.attrs["yunits"] = "Power Spectral Density"   
            
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
