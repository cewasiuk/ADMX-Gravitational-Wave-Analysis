import h5py
from matplotlib import pyplot as plt


h5folder = './hr_data'
h5file = 'run1b_nibble5_spectra.h5'
path = h5folder + '/' + h5file
# Open the HDF5 file in read mode
with h5py.File(path, "r") as file:
    # List all groups (datasets and groups) in the file
    print("Keys in the file:", list(file.keys()))
    
    group_name = str(list(file.keys())[0])
    print(group_name)
    if group_name in file:
        data = file[group_name]
        print(f" \n Length of '{group_name}': ", data )
    print(list(data))

    # yvalues = list(data['measurement_scaled']['yvalues'])
    yvalues = list(data['measurement_filtered']['yvalues'])
    # shape = yvalues.shape
    # dtype = yvalues.dtype
    # print(attrs['xstop'])
    # print(shape)
    # # print(dtype)
    # # Number of data points is the number of bins, i.e. flattened data
    # print('No. of Bins: {}'.format(len(yvalues)))
plt.plot(yvalues, linestyle = 'none', marker = '.')
# plt.ylim(-2,2)
plt.show()