""" Define a vertical integral function """
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
