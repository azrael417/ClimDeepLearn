#!/usr/bin/env python
import netCDF4 as nc
import glob
import datetime as dt
import sys
import os
import scipy.optimize
import numpy as np
import warnings


import scipy.integrate # import the scipy integration library

def calculate_pressure_integral(integrand, pressure):
    """ Calculates the vertical integral (in pressure coordinates) of an array
    
        input:
        ------
            integrand     : the quantity to integrate.  The vertical dimension is assumed to be the first index.
            
            pressure      : the pressure (either a vector or an array of the same shape as integrand).  Units should be [Pa].
            
        output:
        -------
        
            integral      : the approximated integral of integrand (same shape as integrand, but missing the leftmost
                            dimension of integrand).
                            
            For integrand $F(p)$, this function approximates
            
            $$ -\frac{1}{g} \int\limits^{p_s}_0 F(p) dp $$
            
    """
    # set necessary constants
    one_over_negative_g = -1./9.80665 # m/s^2
    
    # determine whether pressure needs to be broadcast to the same shape as integrand
    # check if the dimensions of integrand and pressure don't match
    if not all( [s1 == s2 for s1,s2 in zip(integrand.shape,pressure.shape)] ):
        
        # try broadcasting pressure to the proper shape
        try:
            pressure3d = np.ones(integrand.shape)*pressure[:,np.newaxis,np.newaxis]
        except:
            raise ValueError("pressure cannot be broadcast to the shape of integrand. shape(pressure) = {}, and shape(integrand) = {}".format(pressure.shape,integrand.shape))
    
    # if they do, then simply set pressure3d to pressure
    else:
        pressure3d = pressure
        
    
    # calculate the integral
    # ( fill in any missing values with 0)
    integral = scipy.integrate.simps(np.ma.filled(integrand,0),pressure,axis=0)
    
    # scale the integral and return
    return one_over_negative_g*integral


def create_iwv_ivt_file(merra_file, outfilename = None, do_clobber=False):
    """ Integrates water vapor path and integrated vapor transport from an ARTMIP MERRA file. """

    # set the output file name
    if outfilename is None:
        merra_base = os.path.basename(merra_file).split('.')[0]
        datestr = "_".join(merra_base.split('_')[-2:])

        outfilename = "IWV_IVT_MERRA_{}.nc".format(datestr)

    # check about overwriting
    if os.path.exists(outfilename) and not do_clobber:
        warnings.warn("Skipping creation of {}".format(outfilename))
        return outfilename

    # open the input file
    fin = nc.Dataset(merra_file)

    # read necessary variables
    q = fin.variables['QV'][:]
    u = fin.variables['u'][:]
    v = fin.variables['v'][:]
    pressure = fin.variables['lev'][:]
    lat = fin.variables['lat'][:]
    lon = fin.variables['lon'][:]
    time = fin.variables['time'][:]

    IWV = np.zeros([q.shape[0]] + list(q.shape[2:]))
    IVT = np.zeros([q.shape[0]] + list(q.shape[2:]))

    for t in range(time.shape[0]):
        # calculate IWV [kg/m^2]
        IWV[t,...] = calculate_pressure_integral(q[t,...],pressure*100)

        # calculate IVT [kg/kg/m/s]
        IVT_u = calculate_pressure_integral(q[t,...]*u[t,...],pressure*100)
        IVT_v = calculate_pressure_integral(q[t,...]*v[t,...],pressure*100)
        IVT[t,...] = np.sqrt(IVT_u**2 + IVT_v**2)

    with nc.Dataset(outfilename,'w',clobber=do_clobber,format="NETCDF3_CLASSIC") as fout:
        for dim in ('time','lat','lon'):
            # create the dimension
            if dim == 'time':
                dim_len = None
            else:
                dim_len = len(fin.dimensions[dim])
            fout.createDimension(dim,dim_len)

            # create the dimension variable
            fout.createVariable(dim,fin.variables[dim].dtype,(dim,))

            # copy variable metadata
            for att in fin.variables[dim].ncattrs():
                if att != '_FillValue':
                    fout.variables[dim].setncattr(att,fin.variables[dim].getncattr(att))


        # create the IWV and IVT variables
        vIWV = fout.createVariable('IWV','f4',('time','lat','lon'))
        vIVT = fout.createVariable('IVT','f4',('time','lat','lon'))

        vIWV.long_name = "Integrated Water Vapor Path"
        vIWV.units = "kg m-2"

        vIVT.long_name = "Integrated Vapor Transport"
        vIVT.units = "kg kg-1 m-1 s-1"

        fout.variables['lat'][:] = lat
        fout.variables['lon'][:] = lon
        fout.variables['time'][:] = time

        for t in range(time.shape[0]):
            vIWV[t,...] = IWV
            vIVT[t,...] = IVT

        fin.close()

    return outfilename


if __name__ == "__main__":

    #from simplempi.simpleMPI import simpleMPI as smpi

    #smpi.simpleMPI()

    myinputfiles = sys.argv[1:]

    for inputfile in myinputfiles:
        create_iwv_ivt_file(inputfile,do_clobber = True)




