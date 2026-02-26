import math
import numpy as np
from scipy.stats import ncx2


#make it so the area under a kernel is 1
def normalize_kernel(x):
    mysum=0 
    ret=np.zeros(len(x))
    for i in range(len(x)):
        mysum=mysum+x[i]
    for i in range(len(x)):
        ret[i]=x[i]/mysum
    return ret

#Return dP/dE for maxwell boltzman distribution with a standard deviation E of Estv
#Pro-tip, you can leave Estv and E in units of Hz
#in this case, Estv should be center frequency times beta squared divided by 3
def maxwell_boltzman_distribution(Estv,E):
    return 2*np.sqrt(E/np.pi)*np.exp(-E/Estv)*np.power(Estv,-1.5)

#noncentral chi square distribution, with degree 3 as per wikipedia (since this is what a boosted maxwell boltzman ends up being with k=3)
#v_disp and v_flow are in km/s
#v_disp is square root of the expected value of the axion velocity
#returns
def boosted_maxwell_boltzman_distribution(f,mass_freq,v_disp,v_flow):
    clight=3e5 #speed of light in km/s
    df=3 #degrees of freedom
    #I expect mean to be 3+nc, so the scale of x is
    x0=(0.5*v_disp*v_disp)/(3.0*clight*clight)
    nc=3.0*(v_flow*v_flow)/(v_disp*v_disp) #noncentrality
    #I expect standard deviation to be sqrt(2(3+2*nc))
    x=(f-mass_freq)/(mass_freq*x0)
    dx=1/(mass_freq*x0)
    if f<=mass_freq:
        return 0
    return ncx2.pdf(x,df,nc)*dx


#with this def, mean will be 3
#for a noncentrality of zero, that means the mean chi square is the average energy
# average energy should be 1/2 *

#
def maxwell_boltzman_shape(nbins,binwidth_hz,Eavg_hz):
    sub_bins=10 #This is for integration
    ret=np.zeros(nbins)
    for i in range(nbins):
        int_sum=0
        for j in range(sub_bins):
            df=(float(i))*binwidth_hz+((float(j))*binwidth_hz/float(sub_bins))
            delta=binwidth_hz/sub_bins
            int_sum=int_sum+maxwell_boltzman_distribution(Eavg_hz,df)*(binwidth_hz/float(sub_bins))*delta
        ret[i]=int_sum
    return normalize_kernel(ret)

def boosted_maxwell_boltzman_shape(nbins,binwidth_hz,mass_freq,v_disp,v_flow):
    sub_bins=10 #This is for integration
    ret=np.zeros(nbins)
    for i in range(nbins):
        int_sum=0
        for j in range(sub_bins):
            f=(float(i))*binwidth_hz+((float(j))*binwidth_hz/float(sub_bins))
            df=(binwidth_hz/float(sub_bins))
            int_sum+=boosted_maxwell_boltzman_distribution(f+mass_freq,mass_freq,v_disp,v_flow)*df
        ret[i]=int_sum
    return normalize_kernel(ret)
