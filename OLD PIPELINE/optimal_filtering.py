import math
import numpy as np
import os
import sys
sys.path.insert(0,os.path.abspath('../../datatypes_and_database/'))
sys.path.insert(0,os.path.abspath('../../background_subtraction/sg_background_fit'))
import admx_db_interface
from admx_db_datatypes import PowerSpectrum,PowerMeasurement

#for a given kernel, calculate the least squares estimate of its magnitude in the data
#that is to say that bin j of the output is a_j where 
#minimize sum_i  (input_spectrum[i]-a_j*kernel[i+j])^2/(sigma_i^2)
#input measurement should be an admx measurement class
#kernel should be a numpy array of floats

def deconvolve_with_uncertainty(input_measurement,kernel):
    signal=input_measurement.yvalues
    weights=[1.0/math.pow(input_measurement.yuncertainties[i],2) for i in range(len(input_measurement.yuncertainties)) ]
    # Reverse of kernel -> [::-1]]
    bw_kernel=kernel[::-1]
    sig_kernel=np.convolve(signal*weights,bw_kernel)
    weight_kernel=np.convolve(weights,bw_kernel*bw_kernel)
    best_fit_value=sig_kernel/weight_kernel
    uncertainty_squared=abs(1.0/weight_kernel)
    new_start_freq=input_measurement.xstart-input_measurement.get_xspacing()*(len(kernel)-1)
    new_stop_freq=input_measurement.xstop
    return PowerMeasurement(best_fit_value,np.sqrt(uncertainty_squared),new_start_freq,new_stop_freq,metadata=input_measurement.metadata)

def smooth_with_kernel(input_measurements,kernel,downsample):
    output=deconvolve_with_uncertainty(input_measurements,kernel)
    y_ds=[ output.yvalues[i] for i in range(0,len(output),int(downsample)) ]
    yunc_ds=[ output.yuncertainties[i] for i in range(0,len(output),int(downsample)) ]
    return PowerMeasurement(y_ds,yunc_ds,input_measurements.xstart,input_measurements.xstop)
