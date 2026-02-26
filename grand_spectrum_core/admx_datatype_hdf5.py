import h5py
import math
import datetime
from grand_spectrum_core.admx_db_datatypes import ADMXDataSeries,PowerMeasurement

def save_dataseries_to_hdf5(h5file,location_name,dataseries):
    if location_name in h5file:
        del h5file[location_name]
    grp=h5file.create_group(location_name)
    dset=grp.create_dataset("yvalues",data=dataseries.yvalues)
    grp.attrs['xstart']=dataseries.xstart
    grp.attrs['xstop']=dataseries.xstop
    for key in dataseries.metadata:
        if type(dataseries.metadata[key]) is datetime.datetime:
            grp.attrs[key]=dataseries.metadata[key].isoformat()
        else:
            grp.attrs[key]=dataseries.metadata[key]
    return grp

def load_dataseries_from_hdf5(h5file,location):
    grp=h5file[location]
    metadata={}
    for k in grp.attrs.keys():
        metadata[k]=grp.attrs[k]
    return ADMXDataSeries(grp["yvalues"][:],grp.attrs['xstart'],grp.attrs['xstop'],metadata=metadata)

def save_measurement_to_hdf5(h5file,location_name,measurement):
    if location_name in h5file:
        del h5file[location_name]
    grp=h5file.create_group(location_name)
    dset=grp.create_dataset("yvalues",data=measurement.yvalues)
    dset=grp.create_dataset("yuncertainties",data=measurement.yuncertainties)
    grp.attrs['xstart']=measurement.xstart
    grp.attrs['xstop']=measurement.xstop
    for key in measurement.metadata:
        if type(measurement.metadata[key]) is datetime.datetime:
            grp.attrs[key]=measurement.metadata[key].isoformat()
        else:
            grp.attrs[key]=measurement.metadata[key]
    return grp

def load_measurement_from_hdf5(h5file,location):
    grp=h5file[location]
    metadata={}
    for k in grp.attrs.keys():
        metadata[k]=grp.attrs[k]
    return PowerMeasurement(grp["yvalues"][:],grp["yuncertainties"][:],grp.attrs['xstart'],grp.attrs['xstop'],metadata=metadata)


