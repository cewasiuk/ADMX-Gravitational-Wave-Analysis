import math
import numpy as np
#from erik lentze
#his email says:
#local alpha = 0.3
#  local beta = 1.3
#  local T = 4.7e-7
def pCDM_maxwell_like_form(mass,v,alpha,beta,T):
    """-- mass: axion fundamental mass [eV/c^2]
      -- v: observed RF frequency [Hz]
      -- alpha: polynomial fit parameter []
      -- beta: exponential fit parameter []
      -- T: temperature-like fit parameter []"""

    #--convert frequency, angular frequency --
    speedoflight = 2.998e8  # -- in units m/s
    RME = mass # -- RME = Rest Mass Energy of the axion in eV--
    h = 4.135e-15                           #        --  in units ev s--
    vo = RME/h  #-- rest mass frequency in units of Hz

    gamma = math.gamma((1.0+alpha)/beta)
    Cnum = beta
    Cden = ((1.0/(vo))**beta)**(-(1.0+alpha)/beta)*(1.0/T*vo)**(alpha)*gamma
    C = Cnum/Cden #-- get norm for unit area
    if vo < v :
        A = C*(h*(v-vo)/(mass*T))**(alpha)*math.exp(-(h*(v-vo)/(mass*T))**beta)
    else:
        A = 0
    return A
#

def lentzian_shape(nbins,axionmass_hz,binwidth_hz):
    alpha = 0.3
    beta = 1.3
    T = 4.7e-7
    h = 4.135e-15                                #--  in units ev s--
    sub_bins=10 #This is for integration
    ret=np.zeros(nbins)
    for i in range(nbins):
        int_sum=0
        for j in range(sub_bins):
            df=axionmass_hz+(float(i))*binwidth_hz+((float(j))*binwidth_hz/float(sub_bins))
            int_sum=int_sum+pCDM_maxwell_like_form(axionmass_hz*h,df,alpha,beta,T)*(binwidth_hz/float(sub_bins))

        ret[i]=int_sum
    return ret


