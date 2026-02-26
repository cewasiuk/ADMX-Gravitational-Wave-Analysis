import os
import sys
import math
sys.path.insert(0,os.path.abspath('../../datatypes_and_database/'))
from grand_spectrum_core.admx_db_datatypes import PowerSpectrum,PowerMeasurement
import numpy as np
#from scipy import misc
from scipy.optimize import curve_fit

def chi2_caculator(arr_model, arr_data, uncertainty):
    Ndf = len(np.array(arr_model))
    return 1./Ndf * sum( np.power( ( np.array(arr_model) - np.array(arr_data) )/ np.average( np.array(uncertainty) ) , 2 ))

def mypade(x,a,b,c,d,e,f):
    return (a+b*x+c*x*x+d*x*x*x)/(1+e*x+f*x*x)

def filter_bg_with_pade(input_spectrum):
    """fits background to a form of (A+B*dx+C*dx^2+D*dx3)/(1+E*dx+F*dx2)"""
    deltas=input_spectrum.get_delta_xvalues()
    Q=40000
    scaled_deltas=Q*deltas/((0.5*(input_spectrum.xstart+input_spectrum.xstop)))
    try:
        popt,pcov=curve_fit(mypade,scaled_deltas,input_spectrum.yvalues)
        filtered=np.array( [ mypade(scaled_deltas[i],popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]) for i in range(len(scaled_deltas)) ] )
        ret=PowerSpectrum(input_spectrum.yvalues/filtered,input_spectrum.xstart,input_spectrum.xstop)
        return ret,filtered,popt,ret
    except RuntimeError:
        #print("fit did not converge, pity")
        return input_spectrum,None,[],input_spectrum
    

def filter_bg_with_predefined_filter(input_spectrum,filter_bg):
#    bg = [i / j for i, j in zip(input_spectrum.yvalues, filter_bg.yvalues)]
    ret=PowerSpectrum(input_spectrum.yvalues/filter_bg,input_spectrum.xstart,input_spectrum.xstop)
    filter_bg=np.array(filter_bg)
    return ret,filter_bg

    
def filter_bg_with_pade_plus_signal(input_spectrum,mask=False,**kwargs):
    """fits background to a form of (A+B*dx+C*dx^2+D*dx3)/(1+E*dx+F*dx2)"""
    
    std = np.std(input_spectrum.yvalues)                                                            
    mean = np.mean(input_spectrum.yvalues)                                                          
    diffs = []                                                                                      
    tmp_xvals = []                                                                                  
    tmp_yvals = []                                                                                  
    NbinFromSig = 0                                                                                 

    ## evaluate typical fluctation between next bins.
    for ii in range(1, len(input_spectrum.yvalues)-1):                                              
        diffs.append(abs(input_spectrum.yvalues[ii] - input_spectrum.yvalues[ii-1]))                
    diffs = sorted(diffs)                                                                           
    ## remove outliers which can be signal or background spike.
    del diffs[-10:]                                                                                 

    ## if the difference between the bin and next or next-to-next bin are greater than 
    ## five times mean fluctuation, bins (15 aheads, 3 backwards) are ignored 
    ## from pade filter. 
    for ii in range(len(input_spectrum.yvalues)):                                                   
        if ii<2:                                                                                    
            tmp_xvals.append(input_spectrum.get_xvalues()[ii])                                      
            tmp_yvals.append(input_spectrum.yvalues[ii])                                            
        else:                                                                                       
            tmp_diff1 = abs(input_spectrum.yvalues[ii]-input_spectrum.yvalues[ii-1])                
            tmp_diff2 = abs(input_spectrum.yvalues[ii]-input_spectrum.yvalues[ii-2])                
            if (5*np.mean(diffs) > tmp_diff1 or  5*np.mean(diffs) > tmp_diff2) and NbinFromSig==0:   
                tmp_xvals.append(input_spectrum.get_xvalues()[ii])                                  
                tmp_yvals.append(input_spectrum.yvalues[ii])                                        
            else:                                                                                   
                NbinFromSig += 1                                                                    
            if NbinFromSig ==1:                                                                     
                del tmp_xvals[-3:]                                                                  
                del tmp_yvals[-3:]                                                                  
            if NbinFromSig > 15:                                                                    
                NbinFromSig = 0                                                                     

    if len(tmp_xvals) == 0:
        fity = input_spectrum.yvalues
    else:
        fity = np.interp(input_spectrum.get_xvalues(), tmp_xvals, tmp_yvals)

    
    deltas=input_spectrum.get_delta_xvalues()
    deltas_old = deltas
    fity_old = fity
    if mask:
        pos = kwargs["pos"]
        deltas = deltas[pos]
        fity = fity[pos]

    Q=20000
    scaled_deltas=Q*deltas/((0.5*(input_spectrum.xstart+input_spectrum.xstop)))
    scaled_deltas_old=Q*deltas_old/((0.5*(input_spectrum.xstart+input_spectrum.xstop)))
    try:
        popt,pcov=curve_fit(mypade,scaled_deltas,fity)
        filtered=np.array( [ mypade(scaled_deltas_old[i],*popt) for i in range(len(scaled_deltas_old)) ] )
        ret=PowerSpectrum(input_spectrum.yvalues/filtered,input_spectrum.xstart,input_spectrum.xstop)
        ret_wo_sig=PowerSpectrum(fity_old/filtered,input_spectrum.xstart,input_spectrum.xstop)
        return ret,filtered,popt,ret_wo_sig
    except RuntimeError:
        print("fit did not converge, pity")                                                                                          
        return input_spectrum,None,[],input_spectrum
