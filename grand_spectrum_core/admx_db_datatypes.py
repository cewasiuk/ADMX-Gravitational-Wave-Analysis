import numpy as np
import math

class ADMXDataSeries:
    """This is a series of points y sampled evently between xstart and xstop"""

    def __init__(self,yvalues,xstart,xstop,**kwargs):
        self.xstart=xstart
        self.xstop=xstop
        self.yvalues=yvalues
        self.xunits="Unknown"
        self.yunits="Unknown"
        self.metadata={}
        if "metadata" in kwargs:
            self.metadata=kwargs["metadata"]

    def copy(self):
        return ADMXDataSeries(np.copy(self.yvalues),self.xstart,self.xstop,metadata=self.metadata)

    def get_xvalues(self):
        """Get the x values as an array"""
        return np.linspace(self.xstart,self.xstop,len(self.yvalues))

    def get_xspacing(self):
        """Gets the spacing between two x values"""
        return (self.xstop-self.xstart)/len(self.yvalues)

    def get_delta_xvalues(self):
        """Get the offset of x values from the array center as an array"""
        df=self.xstop-self.xstart
        return np.linspace(-df/2,df/2,len(self.yvalues))

    def subseries(self,starti,endi):
        return ADMXDataSeries(self.yvalues[starti:endi],self.get_x_at_index(starti),self.get_x_at_index(endi),metadata=self.metadata)

    def __str__(self):
        """Print out the data as a string"""
        ret=""
        xs=self.get_xvalues()
        ret=ret+"#\"{}\" \"{}\"\n".format(self.xunits,self.yunits)
        for i in range(len(self.yvalues)):
            ret=ret+"{} {}\n".format(xs[i],self.yvalues[i])
        return ret

    def __len__(self):
        return len(self.yvalues)

    def get_x_index_below_x(self,xval):
        """Returns the x index closest to xval on the low side"""
        return int(math.floor(float(len(self.yvalues))*(xval-self.xstart)/(self.xstop-self.xstart)))

    def get_x_at_index(self,index):
        return self.xstart+float(index)*(self.xstop-self.xstart)/float(len(self.yvalues))

    def interp_y_at_x(self,xval):
        """Returns y at x, interpolating between closest points, clip to edges"""
        lowbin=self.get_x_index_below_x(xval)
        if lowbin<0:
            return self.yvalues[0]
        if lowbin>=len(self.yvalues):
            return self.yvalues[-1]
        x1=self.get_x_at_index(lowbin)
        x2=self.get_x_at_index(lowbin+1)
        y1=self.yvalues[lowbin]
        y2=self.yvalues[lowbin+1]
        return y1+(y2-y1)*(x-x1)/(x2-x1)

    def __add__(self,toadd):
        if isinstance(toadd,ADMXDataSeries):
            #Should I check lengths and units here?
            return ADMXDataSeries(self.yvalues+toadd.yvalues,self.xstart,self.xstop,metadata=self.metadata)
        raise Exception('Adding something to a data series that is not handled')
 
    def __truediv__(self,todiv):
        if isinstance(todiv,np.ndarray) or isinstance(todiv,float):
            return ADMXDataSeries(self.yvalues/todiv,self.xstart,self.xstop,metadata=self.metadata)
        raise Exception('Dividing something to a data series that is not handled')

class PowerSpectrum(ADMXDataSeries):

    """This is a spectrum of power vs frequency, defaults to Watts vs MHz"""
    def __init__(self,power,start_freq,stop_freq,**kwargs):
        ADMXDataSeries.__init__(self,power,start_freq,stop_freq,**kwargs)
        self.xunits="Frequency (MHz)"
        self.yunits="Power (Watts)"

class PowerMeasurement(PowerSpectrum):
    """This is a spectrum of power vs time, defaults to Watts vs MHz
        with the inclusion of uncertaintie in the yuncertainties array"""
    def __init__(self,power,power_unc,start_freq,stop_freq,**kwargs):
        PowerSpectrum.__init__(self,power,start_freq,stop_freq,**kwargs)
        self.yuncertainties=power_unc

    def copy(self):
        return PowerMeasurement(np.copy(self.yvalues),np.copy(self.yuncertainties),self.xstart,self.xstop,metadata=self.metadata)

    def subspectrum(self,starti,endi):
        return PowerMeasurement(self.yvalues[starti:endi],self.yuncertainties[starti:endi],self.get_x_at_index(starti),self.get_x_at_index(endi))

    def __truediv__(self,todiv):
        if isinstance(todiv,np.ndarray) or isinstance(todiv,float):
            return PowerMeasurement(self.yvalues/todiv,self.yuncertainties/todiv,self.xstart,self.xstop,metadata=self.metadata)
        raise Exception('Dividing something to a data series that is not handled')

    def __mul__(self,tomul):
        if isinstance(tomul,np.ndarray):
            if len(self) != len(tomul):
                raise Exception("trying to multiply by array of different length")
        if isinstance(tomul,np.ndarray) or isinstance(tomul,float):
            #print("X")
            # print(self.yvalues)
            # print("Y")
            # print(tomul)
            ## print("Z")
            # print(self.yvalues*tomul)
            return PowerMeasurement(self.yvalues*tomul,self.yuncertainties*tomul,self.xstart,self.xstop,metadata=self.metadata)
        raise Exception('Multiplying something to a data series that is not handled',type(tomul))

    def update_bin_with_additional_measurement(self,binno,p,dp):
        p1=self.yvalues[binno]
        w1=1.0/(self.yuncertainties[binno]*self.yuncertainties[binno])
        p2=p
        w2=1.0/(dp*dp)
        pnew=(p1*w1+p2*w2)/(w1+w2)
        wnew=w1+w2
        self.yvalues[binno]=pnew
        self.yuncertainties[binno]=math.sqrt(1.0/wnew)

    def update_with_additional_measurement(self,other):
        """updates with another measurement.  hope you're ok with half bin misalignment"""
        myistart=max(0,self.get_x_index_below_x(other.xstart))
        myistop=min(len(self),self.get_x_index_below_x(other.xstop))
        for i in range(myistart,myistop):
            thex=self.get_x_at_index(i)
            j=other.get_x_index_below_x(thex)
            self.update_bin_with_additional_measurement(i,other.yvalues[j],other.yuncertainties[j])

    def __str__(self):
        """Print out the data as a string"""
        ret=""
        xs=self.get_xvalues()
        ret=ret+"#\"{}\" \"{}\" \"{}\"\n".format(self.xunits,self.yunits,"uncertainty")
        for i in range(len(self.yvalues)):
            ret=ret+"{} {} {}\n".format(xs[i],self.yvalues[i],self.yuncertainties[i])
        return ret


