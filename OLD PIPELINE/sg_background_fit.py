import os
import sys
import math
sys.path.insert(0,os.path.abspath('../../datatypes_and_database/'))
from admx_db_datatypes import PowerSpectrum,PowerMeasurement
import numpy as np
from scipy.signal import savgol_filter

def filter_bg_with_sg(input_spectrum,sg_polyorder,sg_window_length):
    """Applies a savitsky golay filter to the input spectrum
       returns the filtered spectrum, and an array that describes the filter shape"""
    filtered=savgol_filter(input_spectrum.yvalues,sg_window_length,sg_polyorder,mode='interp')
    ret=PowerSpectrum(input_spectrum.yvalues/filtered,input_spectrum.xstart,input_spectrum.xstop)
    return ret,filtered

def filter_bg_with_sg_keep_extrema(input_spectrum,sg_polyorder,sg_window_length,extrema_fraction):
    """Applies a savitsky golay filter to the input spectrum
       then exise the most extreme bins (as given by extrema fraction) and refit to allow more extrema through
       returns the filtered spectrum, and an array that describes the filter shape"""
    inputcopy=np.array(input_spectrum.yvalues)
    filtered=savgol_filter(input_spectrum.yvalues,sg_window_length,sg_polyorder,mode='interp')
    firstpassvalues=input_spectrum.yvalues/filtered
    highest=np.argsort(abs(firstpassvalues))
    extrema_count=math.floor(extrema_fraction*len(input_spectrum))
    for i in range(extrema_count):
        inputcopy[ highest[-i-1] ]=filtered[ highest[-i-1] ]
    filtered=savgol_filter(inputcopy,sg_window_length,sg_polyorder,mode='interp')
    ret=PowerSpectrum(input_spectrum.yvalues/filtered,input_spectrum.xstart,input_spectrum.xstop)
    return ret,filtered
        
