
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import scipy.signal as sig


# In[1]:


def noisyrk4(s,t,tau,derivsRK,i,vN0,vN1,vN2):
    """modified RK4 integrator including noise
    DEPENDENCIES
        none
    INPUTS
        s - initial state vector [delta]
        t - time
        tau - time step size
        derivsRK - RHS of ODE, fn defined somewhere
        i - bias current
        vN0,vN1,vN2 - voltage noise values at t=t, t=t+tau/2, t=t+tau
    OUTPUTS
        sout - new state vector [delta]"""
    
    half_tau = 0.5*tau
    
    F1 = derivsRK(s,t,i,vN0)                    # use current voltage noise
    t_half = t + half_tau
    stemp = s + half_tau*F1
    
    F2 = derivsRK(stemp,t_half,i,vN1)           # use half-tau step voltage noise
    stemp = s + half_tau*F2
    
    F3 = derivsRK(stemp,t_half,i,vN1)           # use half-tau step voltage noise
    t_full = t + tau
    stemp = s + tau*F3
    
    F4 = derivsRK(stemp,t_full,i,vN2)           # use full-tau step voltage noise
    sout = s + tau/6.*(F1 + F4 + 2.*(F2 + F3))
    return sout   


# In[3]:


def snJJRK2(s,t,i,vN):
    """Returns RHS of ODE representing single Josephson junction
    DEPNDENCIES
        numpy as np
    INPUTS
        s - state vector [delta]
        t - time
        i - bias current
        vN - voltage noise value"""

    deriv = i - np.sin(s) + vN                  # d(del)/dtheta = v = i - sin(del) + vN
    return(deriv)


# In[4]:


def snJJ2(nStep,tau,s,i,Gamma):
    """hanlder function for RK4 solver for noisy single Josephson junction
    DEPENDENCIES
        noisyrk4()
        numpy as np
    INPUTS
        nStep - number of time steps to simulate
        tau - time step size
        s - initial state vector [delta]
        i - bias current
        Gamma - Johnson noise parameter"""
    
    var = 4*Gamma/tau                                   # var = 2*Gamma/tau
    sd = var**.5                                        # std dev = var^.5, for gaussian noise
    theta = 0                                           # set time theta to zero
    vN = np.zeros(2*nStep+1)                            # crate holder for noise values
    for N in range(2*nStep+1): 
        vN[N] = np.random.normal(0,sd)                  # fill noise values, two for each tau
    X = np.zeros([3,nStep])                             # create output array
    
    X[0,0] = theta                                      # record initial time theta
    X[1,0] = s                                          # record initial del, phase diff
    X[2,0] = i - np.sin(s) + vN[0]                      # record initial voltage
    
    for iStep in range(1,nStep):                        # for loop through time
        vN0 = vN[2*iStep-2]  
        vN1 = vN[2*iStep-1]                               
        vN2 = vN[2*iStep]                               # read off noise values to use

        s = noisyrk4(s,theta,tau,snJJRK2,i,vN0,vN1,vN2) # call rk4 with snJJRK to get next state value
        
        X[0,iStep] = theta                              ##  theta  ##   record time theta to output array
        X[1,iStep] = s                                  ##  delta  ##   record state, phase diff
        X[2,iStep] = i - np.sin(s) + vN2                ## voltage ##   record v, i - sin(delta) + vn
        theta = theta + tau                             # advance time for start of next cycle
        
    return(vN[0:-1:2],X)                                # return every other noise voltage value

