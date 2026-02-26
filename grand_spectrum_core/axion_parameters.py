#TODO clean these up, they're a mess and should use constants from scipy instead of the hard coded ones
import math

#as per Daw thesis page 24
#V is in m^3
#B is in T
#ff is unitless form factor
#f in is Hz
#Q is cavity Q
#KSVZ coupling assumed
#dark matter density of 0.45 GeV/cc assumed
def ksvz_axion_power(bsquared_v_ff,f,Q):
    return 1.52e-21*bsquared_v_ff*(1000./220.)*math.pow(1./7.6,2.)*(Q/70000.)*(f/(750.*1e6))

def dfsz_axion_power(bsquared_v_ff,f,Q):
    return ksvz_axion_power(bsquared_v_ff,f,Q)*(0.36*0.36)/(0.97*0.97) #TODO double check 

#f is in Hz
def ksvz_coupling(f):
    fsc=7.29e-3
    ggammaksvz=0.97
#hbar=6.58e-22*1e6 #in eV s
    h=4.134e-15

    return 1e-7*(f*h/0.62)*(fsc*ggammaksvz)/(3.14)

#f is in Hz
def dfsz_coupling(f):
    fsc=7.29e-3
    #ggammaksvz=0.97
    ggammadfsz=0.36
#hbar=6.58e-22*1e6 #in eV s
    h=4.134e-15

    return 1e-7*(f*h/0.62)*(fsc*ggammadfsz)/(3.14)

