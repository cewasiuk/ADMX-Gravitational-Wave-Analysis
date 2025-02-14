import math
import numpy as np
from scipy.stats import ncx2
import decimal

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
def turner_axion_distribution(m_a,E,t,turner_parameters):
    #in km
    c_light=3e5
    v_halo=turner_parameters["v_halo"]#29.8
    v_sun=turner_parameters["v_sun"]
    v_earth=turner_parameters["v_earth"]#29.8
    v_rot=turner_parameters["v_rot"]#0.465
    v_halo=v_halo/c_light
    v_sun=v_sun/c_light
    v_earth=v_earth/c_light
    v_rot=v_rot/c_light
    #Seattle is at the 49th parallel
    l=turner_parameters["parallel"]*math.pi/180
    #definition of beta
    beta=1.5/(v_halo**2)
    #Angle of the Sun's velocity above the ecliptic 
    alpha=turner_parameters["alpha"]*math.pi/180
    #Angle between the Sun's velocity and the equatorial plane
    psi=turner_parameters["psi"]*math.pi/180
    #Earth's orbital angular velocity
    omega_o=turner_parameters["omega_o"]#1.991e-7
    #Earth's rotational angular velocity
    omega_r=turner_parameters["omega_r"]#7.292e-5
    v_e=v_sun*math.sqrt(1+(2*v_earth/v_sun)*(math.cos(alpha)*math.cos(omega_o*t))+(v_rot/v_sun)*(math.cos(l)*math.cos(psi)*math.cos(omega_r*t)))

    return 2.0*math.sqrt(beta/math.pi)*(1/(m_a*v_e))*np.exp(-beta*v_e*v_e-2*beta*E/m_a)*np.sinh(2*beta*np.sqrt(2*E/m_a)*v_e)


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
def turner_axion_shape(nbins,binwidth_hz,Eavg_hz,elapsed_time,turner_parameters):
    sub_bins=10 #This is for integration
    ret=np.zeros(nbins)
    for i in range(nbins):
        int_sum=0
        for j in range(sub_bins):
            df=(float(i))*binwidth_hz+((float(j))*binwidth_hz/float(sub_bins))
            delta=binwidth_hz/sub_bins
            int_sum=int_sum+turner_axion_distribution(Eavg_hz,df,elapsed_time,turner_parameters)*(binwidth_hz/float(sub_bins))*delta
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
