import sys
import os
import numpy as np
sys.path.insert(0,os.path.abspath('../../datatypes_and_database/'))
from admx_db_datatypes import PowerMeasurement,ADMXDataSeries


#Keeping track of sum(1/sigma^2),   sum(x/sigma^2), sum(x^2/sigma^2) for each bin
#update these spectra with a single new spectrum
class ADMXGrandSpectrum:
    def __init__(self,nbins,start_freq,stop_freq):
        epsilon_size=1e-24 #this is the default uncertainty for an unmeasured bin : WHY 
        self.sum_weights=ADMXDataSeries(np.ones(nbins)*epsilon_size,start_freq,stop_freq)
        self.sum_x_weights=ADMXDataSeries(np.zeros(nbins),start_freq,stop_freq)
        self.sum_xx_weights=ADMXDataSeries(np.zeros(nbins),start_freq,stop_freq)
        self.counts_array=np.zeros(nbins)
        self.start_freq=start_freq
        self.stop_freq=stop_freq

    def update_bin_with_additional_measurement(self,binno,p,dp):
        self.counts_array[binno]+=1
        w=1.0/(dp*dp)
        self.sum_weights.yvalues[binno]+=w
        self.sum_x_weights.yvalues[binno]+=p*w
        self.sum_xx_weights.yvalues[binno]+=p*p*w

    def update_with_additional_measurement(self,other):
        """updates with another measurement.  hope you're ok with half bin misalignment"""
        myistart=max(0,self.sum_weights.get_x_index_below_x(other.xstart))
        myistop=min(len(self.sum_weights),self.sum_weights.get_x_index_below_x(other.xstop))
        for i in range(myistart,myistop):
            thex=self.sum_weights.get_x_at_index(i)
            j=other.get_x_index_below_x(thex)
            self.update_bin_with_additional_measurement(i,other.yvalues[j],other.yuncertainties[j])

    def get_average_power_spectrum(self):
        #this returns the spectrum of best estimate of power with uncertainty at each frequency
        #assuming each contributed measurement was of the same thing (i.e., assuming the power is not fluctuating over time)
        epsilon_size=1e-24
        ret=PowerMeasurement(self.sum_x_weights.yvalues/self.sum_weights.yvalues,np.sqrt(1.0/self.sum_weights.yvalues),self.start_freq,self.stop_freq)
        return ret

    def get_chisq_spectrum(self):
        #this returns the chi-square over number of samples for each bin
        #A value near 1 indicates that it is consistent with measuring the same thing over and over (usually zero)
        #A value much greater than one indicates that it is more likely that the power is fluctuating between samples. 
        avg_power=self.get_average_power_spectrum()
        values=np.array([ self.sum_weights.yvalues[i]*avg_power.yvalues[i]**2-2*self.sum_x_weights.yvalues[i]*avg_power.yvalues[i]+self.sum_xx_weights.yvalues[i]  for i in range(len(self.sum_weights)) ])
        return ADMXDataSeries(values/self.counts_array,self.start_freq,self.stop_freq)

    def get_contributing_counts(self):
        return ADMXDataSeries(self.counts_array,self.start_freq,self.stop_freq)

