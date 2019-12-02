
# coding: utf-8

# In[14]:


import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import scipy.signal as sig
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy import interpolate
import csv
import datetime


# In[4]:


import time, sys
from IPython.display import clear_output

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)


# In[5]:


def qSquidRK(s,th,param):    
    """Returns derivatives of the SQUID system
    DEPENDENCIES
        numpy as np
    INPUTS
        s - state vector, [delta_1, delta_2]
        th - dimensioless time
        par - array
            [alpha,betaL,eta,rho,i,phia]
            alpha - critical current symmetry parameter (0 to 1)
            beta - inductance constant
            eta - inductance symmetry parameter (0 to 1)
            rho - resistance symmetry parameter (0 to 1)
            i - dimensionless bias current
            phia - dimensionless applied flux
    OUTPUTS
        deriv - array of derivs, [ddelta_1/dth, ddelta_2/dth,d^2delta_1,d^2delta_2]"""
    
    alpha = param[0]; beta = param[1];
    eta = param[2]; rho = param[3]
    i = param[4]; phia = param[5]
    del1 = s[0] # del_1(theta)
    del2 = s[1] # del_2(theta)
    j = (del1 - del2 - 2*np.pi*phia)/(np.pi*beta)- eta*i/2
    
    d1 = (.5*i-j-(1-alpha)*np.sin(del1))/(1-rho)
    d2 = (.5*i+j-(1+alpha)*np.sin(del2))/(1+rho)
    
    deriv = np.array([d1,d2])
    return(deriv)


# In[6]:


def rk4(x,t,tau,derivsRK,param):
    """RK4 integrator modified to use noise
    INPUTS
        s - state vector
        th - time, theta
        tau - time step size
        derivsRK - RHS of ODE, fn defined somewhere
        par - array
            [alpha,betaL,eta,rho,i,phia]
    OUTPUTS
        sout - new state vector new time
            [delta_1,delta_2,ddelta_1,ddelta_2]"""

    half_tau = 0.5*tau
    
    F1 = derivsRK(x,t,param)
    t_half = t + half_tau
    xtemp = x + half_tau*F1
    
    F2 = derivsRK(xtemp,t_half,param)
    xtemp = x + half_tau*F2
    
    F3 = derivsRK(xtemp,t_half,param)
    t_full = t + tau
    xtemp = x + tau*F3
    
    F4 = derivsRK(xtemp,t_full,param)
    xout = x + tau/6.*(F1 + F4 + 2.*(F2 + F3))
    return xout


# In[7]:


def qSQUID(nStep,tau,s,par):
    """Handles RK4 solver, returns time series sim of SQUID state vector
        including voltage time series
    DEPENDENCIES
        numpy as np
        rk4() - basic Runge-Kutta 4th order solver
    INPUTS
        par - parameter vector
            [alpha, beta, eta, rho, i, phia]
        s - state vector [delta_1(t=0), delta_2(t=0)]
        tau - time step size
        nStep - number of time steps to simulate
    OUTPUT
        X - time series of state vector
            [theta,delta_1,delta_2,j,ddel1/dth,ddel2/dth,v]"""
    theta = 0
    X = np.zeros([7,nStep])
    X[1,0] = s[0]
    X[2,0] = s[1]
    X[3,0] = (s[0] - s[1] - 2*np.pi*par[5])/(np.pi*par[1]) - par[2]*par[4]/2
    # j0 = (del10 - del20 - 2*np.pi*phia)/(np.pi*beta) - eta*i/2
    X[4,0] = (par[4]/2 - X[3,0] -(1-par[0])*np.sin(s[0]))/(1-par[3])
    # (i/2 - j0 -(1-alpha)*np.sin(del10))/(1-rho)
    X[5,0] = (par[4]/2 + X[3,0] -(1-par[0])*np.sin(s[1]))/(1+par[3])
    # (i/2 + j0 -(1-alpha)*np.sin(del20))/(1+rho)
    X[6,0] = (1+par[2])*X[4,0]/2 + (1-par[2])*X[5,0]/2
    # (1+eta)*d10/2 + (1-eta)*d20/2
    
    for iStep in range(nStep):
        
        s = rk4(s, theta, tau, qSquidRK, par)
        
        X[0,iStep] = theta
        X[1,iStep] = s[0]
        X[2,iStep] = s[1]
        X[3,iStep] = (s[0] - s[1] - 2*np.pi*par[5])/(np.pi*par[1]) - par[2]*par[4]/2
        X[4,iStep] = (.5*par[4]-X[3,iStep]-(1-par[0])*np.sin(s[0]))/(1-par[3])
        X[5,iStep] = (.5*par[4]+X[3,iStep]-(1+par[0])*np.sin(s[1]))/(1+par[3])
        X[6,iStep] = (1+par[2])*X[4,iStep]/2 + (1-par[2])*X[5,iStep]/2
        
        theta = theta + tau
        
    return(X)


# In[8]:


def vj_timeseries(nStep,tau,s,par):
    """Returns time series simulation of squid, figure and csv
    DEPENDENCIES
    qSQUID()
        numpy as np
        matplotlib.pyplot as plt
    INPUTS
        nStep - number of steps to run in time series
        tau - step size for time series
        s - initial state vector [delta_1[theta=0],delta_2[theta=0]]
        par - parameter vector
            [alpha, beta, eta, rho, i, phia]
    OUTPUTS
        figure - plots of
            voltage time series w average
            circulating current time series w average
            output to screen
            png 'timeseriesdatetime.png' saved to parent directory
        csv - time series csv file containing
            theta,delta_1,delta_2,j,ddel1/dth,ddel2/dth,v
            csv 'timeseriesdatetime.csv' saved to parent directory            
        """
    # run sim
    S = qSQUID(nStep,tau,s,par)
    # chop off first 10% of time series to remove any transient
    md = int(.1*len(S[0,:]))
    
    # build figure title with parameters used
    ti = r'$\alpha$=%s, $\beta_L$=%s, $\eta$=%s'% (str(round(par[0],3)),str(round(par[1],3)),str(round(par[2],3)))+'\n'+            r'$\rho$=%s, $i$=%s, $\phi_a$=%s' % (str(round(par[3],3)),str(round(par[4],3)),str(round(par[5],3)))
    
    # plot
    fig, ax = plt.subplots(2,1,figsize=(3,5))
    fig.suptitle(ti)
    ax1 = plt.subplot(2,1,1)
    ax1.plot(S[0,md:],S[6,md:])
    ax1.hlines((sum(S[6,md:])/len(S[6,md:])),S[0,md],S[0,-1],linestyle='dotted')
    ax1.set(ylabel="Voltage, v",
       xticklabels=([]))
    ax2 = plt.subplot(2,1,2)
    ax2.plot(S[0,md:],S[3,md:])
    ax2.hlines((sum(S[3,md:])/len(S[3,md:])),S[0,md],S[0,-1],linestyle='dotted')
    ax2.set(ylabel="Circ Current, j",
       xlabel=r"Time,$\theta$")
    
    # create output file metadata    
    meta1 = ['# alpha=%s'%par[0],'beta=%s'%par[1],'eta=%s'%par[2],'rho=%s'%par[3],'phia=%s'%par[4]]
    meta2 = ['# nStep=%s'%nStep,'tau=%s'%tau]
    header = ['theta','delta_1','delta_2','j','ddel1/dth','ddel2/dth','v']
    csvtime = datetime.datetime.now()
    timestr = [datetime.datetime.strftime(csvtime, '# %Y/%m/%d, %H:%M:%S')]
    timeti = str(datetime.datetime.strftime(csvtime, '%Y%m%d%H%M%S'))
    csvtitle='timeseries'+timeti+'.csv'
    pngtitle='timeseris'+timeti+'.png'
    Sf = np.matrix.transpose(S)
    
    # create, write, output(close) csv file
    with open(csvtitle, 'w') as csvFile:
        filewr = csv.writer(csvFile,delimiter=',')
        filewr.writerow(timestr)
        filewr.writerow(meta1)
        filewr.writerow(meta2)
        filewr.writerow(header)
        filewr.writerows(Sf)
    csvFile.close()
    # save figure
    fig.savefig(pngtitle)
    print('csv file written out:', csvtitle)
    print('png file written out:', pngtitle)


# In[9]:


def iv_curve(nStep,tau,s,par,alpha=0,beta_L=0,eta=0,rho=0,phia=0):
    """Returns contour plot and data file for IV curves
    DEPENDENCIES
        qSQUID()
        update_progress()
        numpy as np
        matplotlib.pyplot as plt
    INPUTS
        nStep - number of steps to run in time series
        tau - step size for time series
        s - initial state vector [delta_1[theta=0],delta_2[theta=0]]
        par - parameter vector
            [alpha, beta_L, eta, rho, i, phia]
        input parameter LIST - alpha, beta, eta, rho, phia
            multiple values of input parameter as list
            draws contour for each
            if given, overwrites value in par
            if not given, value from par is used for one contour
            ONLY SUPPLY maximum of one input list here
    OUTPUTS
        plot - IV contours at levels given in input param array
            output to screen
            png 'IVdatetime.png' saved to parent directory
        csv - IV contours at levels given
            csv 'IVdatetime.png' saved to parent directory
        """
    # create currents to sweep
    i = np.arange(0.,6.,.1)
    
    ch = 0 # check for only one parameter sweeped.
    k = 1 # set 0 axis dim to 1 at min
    md = int(0.1*len(i)) # cut of the first 10 percent of points in time series
    
    # check if an array was given for an input parameter
    # k - length of input parameter array (number of contours)
    # parj - build a list of parameters to pass at each array value of that parameter
    # la, lc - plot label and csv header lable
    # lb - rename parameter array to add in plot and header later
    # ti - plot title
    # meta1 - csv metadata
    # ch - check value, check for only one input parameter array, or none for one contour
    if alpha != 0:
        alpha = np.array(alpha)
        k = len(alpha)
        parj = np.zeros([k,6])
        la = r'$\alpha$'; lc = 'alpha'
        lb = np.copy(alpha)
        ti = r'$\beta_L$=%s, $\eta$=%s, $\rho$=%s, $\phi_a$=%s' % (str(round(par[1],3)),str(round(par[2],3)),str(round(par[3],3)),str(round(par[5],3)))
        meta1 = ['# beta_L=%s'%par[1],'eta=%s'%par[2],'rho=%s'%par[3],'phi_a=%s'%par[5]]
        # add input array values to iteration parameters as appropriate
        for j in range(k):
            parj[j,:] = np.array([alpha[j],par[1],par[2],par[3],0.,par[5]])
        ch = ch + 1
    if beta_L != 0:
        beta_L = np.array(beta_L)
        k = len(beta_L)
        parj = np.zeros([k,6])
        la = r'$\beta_L$'; lc = 'beta_L'
        lb = np.copy(beta_L)
        ti = r'$\alpha$=%s, $\eta_L$=%s, $\rho$=%s, $\phi_a$=%s' % (str(round(par[0],3)),str(round(par[2],3)),str(round(par[3],3)),str(round(par[5],3)))
        meta1 = ['# alpha=%s'%par[0],'eta_L=%s'%par[2],'rho=%s'%par[3],'phi_a=%s'%par[5]]
        for j in range(k):
            parj[j,:] = np.array([par[0],beta[j],par[2],par[3],0.,par[5]])
        ch = ch + 1
    if eta != 0:
        eta = np.array(eta)
        k = len(eta)
        parj = np.zeros([k,6])
        la = r'$\eta$'; lc = 'eta'
        lb = np.copy(eta)
        ti = r'$\alpha$=%s, $\beta_L$=%s, $\rho$=%s, $\phi_a$=%s' % (str(round(par[0],3)),str(round(par[1],3)),str(round(par[3],3)),str(round(par[5],3)))
        meta1 = ['# alpha=%s'%par[0],'beta_L=%s'%par[1],'rho=%s'%par[3],'phi_a=%s'%par[5]]
        for j in range(k):
            parj[j,:] = np.array([par[0],par[1],eta[j],par[3],0.,par[5]])
        ch = ch + 1
    if rho != 0:
        rho = np.array(rho)
        k = len(rho)
        parj = np.zeros([k,6])
        la = r'$\rho$'; lc = 'rho'
        lb = np.copy(phia)
        ti = r'$\alpha$=%s, $\beta_L$=%s, $\eta$=%s, $\phi_a$=%s' % (str(round(par[0],3)),str(round(par[1],3)),str(round(par[2],3)),str(round(par[5],3)))
        meta1 = ['# alpha=%s'%par[0],'beta_L=%s'%par[1],'eta=%s'%par[2],'phi_a=%s'%par[5]]
        for j in range(k):
            parj[j,:] = np.array([par[0],par[1],par[2],rho[j],0.,par[5]])
        ch = ch + 1
    if phia != 0:
        phia = np.array(phia)
        k = len(phia)
        parj = np.zeros([k,6])
        la = r'$\phi_a$'; lc = 'phi_a'
        lb = np.copy(phia)
        ti = r'$\alpha$=%s, $\beta_L$=%s, $\eta$=%s, $\rho$=%s' % (str(round(par[0],3)),str(round(par[1],3)),str(round(par[2],3)),str(round(par[3],3)))
        meta1 = ['# alpha=%s'%par[0],'beta_L=%s'%par[1],'eta=%s'%par[2],'rho=%s'%par[3]]
        for j in range(k):
            parj[j,:] = np.array([par[0],par[1],par[2],par[3],0.,phia[j]])
        ch = ch + 1
    # if check value is more than one, too many input parameter arrays given
    if ch > 1:
        return('Please supply at most one parameter to sweep')
    # if check value zero, assume plotting only one contour
    if ch == 0:
        parj = np.zeros([2,6])
        parj[0,:] = par
        parj[1,:] = par
        ti = r'$\alpha$=%s, $\beta_L$=%s, $\eta$=%s, $\rho$=%s, $\phi_a$=%s' % (str(round(par[0],3)),str(round(par[1],3)),str(round(par[2],3)),str(round(par[3],3)),str(round(par[5],3)))
        meta1 = ['# alpha=%s'%par[0],'beta_L=%s'%par[1],'eta=%s'%par[2],'rho=%s'%par[3],'phi_a=%s'%par[5]]
    # build sim output array of appropriate size
    # needs as many rows as contours determined by input parameter array
    if k > 1:
        V = np.zeros([k,len(i)])
    else:
        V = np.zeros([2,len(i)])
        
    # cp - check progress, total outputs in V
    cp = k*len(i)
    
    # loop over k rows and len(i) colums of V
    # fill V with average voltage from time series for given params
    # parjj - parameter array for this time series
    # S - state array output from sim
    for j in range(k):
        parjj = parj[j,:]
        for m in range(len(i)):
            parjj[4] = i[m]
            S = qSQUID(nStep,tau,s,parjj)
            V[j,m] = sum(S[6,md:])/len(S[6,md:])
            # new progress bar current iter/total iters
            update_progress((m + j*len(i))/cp)
    # fill out progress bar
    update_progress(1)
    
    # build output for csv
    # join i values and average Voltage matrix
    Sf = np.concatenate((np.matrix(i),V),axis=0)
    # flip independent axis, i, from horizontal to vertical
    Sf = np.matrix.transpose(Sf)
    # convert from matrix to array to ease csv output
    Sf = np.array(Sf)
    
    # make a figure
    # header - csv header info, param input value for contour
    fig,ax = plt.subplots()
    # one contour, or
    if k == 1:
        ax.plot(V[0],i)
        header = ['i','V']
    # k contours
    else:
        header = ['i']*(k+1)
        for j in range(k):
            ax.plot(V[j],i,label= la + '=%s' % str(round(lb[j],3)))
            header[j+1] = lc + '=%s' % str(round(lb[j],3))
    # ic = 0 line for comparison
    ax.plot(np.arange(0,2.6,.1),np.arange(0,5.2,.2),'--',
       label=r"$i_c=0$")
    ax.set(title=ti,
       xlabel=r"Average voltage, $\bar{v}$",
       ylabel="Bias current, i",
       xlim=[0,2.5],ylim=[0,6.])
    ax.legend()
    fig.tight_layout()
    
    # build rest of metadata needed for csv
    meta2 = ['# nStep=%s'%nStep,'tau=%s'%tau]
    csvtime = datetime.datetime.now()
    timestr = [datetime.datetime.strftime(csvtime, '# %Y/%m/%d, %H:%M:%S')]
    timeti = str(datetime.datetime.strftime(csvtime, '%Y%m%d%H%M%S'))
    csvtitle='IV'+timeti+'.csv'
    pngtitle='IV'+timeti+'.png'
    
    # create, write, and save(close) csv
    with open(csvtitle, 'w') as csvFile:
        filewr = csv.writer(csvFile,delimiter=',')
        filewr.writerow(timestr)
        filewr.writerow(meta1)
        filewr.writerow(meta2)
        filewr.writerow(header)
        filewr.writerows(Sf)
    csvFile.close()
    # save figure
    fig.savefig(pngtitle)
    print('csv file written out:', csvtitle)
    print('png file written out:', pngtitle)


# In[10]:


def vphi_curve(nStep,tau,s,par,alpha=0,beta_L=0,eta=0,rho=0,i=0):
    """Returns contour plot and data file for IV curves
    DEPENDENCIES
        qSQUID()
        update_progress()
        numpy as np
        matplotlib.pyplot as plt
    INPUTS
        nStep - number of steps to run in time series
        tau - step size for time series
        s - initial state vector [delta_1[theta=0],delta_2[theta=0]]
        par - parameter vector
            [alpha, beta, eta, rho, i, phia]
        input parameter array - alpha, beta, eta, rho, phia
            multiple values of input parameter as array
            draws contour for each
            if given, overwrites value in par
            if not given, value from par is used for one contour
            ONLY SUPPLY maximum of one input array here
    OUTPUTS
        plot - IV contours at levels given in input param array
            output to screen
            png 'IVdatetime.png' saved to parent directory
        csv - IV contours at levels given
            csv 'IVdatetime.png' saved to parent directory
        """
    # create fluxes to sweep
    phia = np.arange(0.,1.1,.1)
    
    ch = 0 # check for only one parameter sweeped.
    k = 1 # set 0 axis dim to 1 at min
    md = int(0.1*len(phia)) # cut of the first 10 percent of points in time series
    
    # check if an array was given for an input parameter
    # k - length of input parameter array (number of contours)
    # parj - build a list of parameters to pass at each array value of that parameter
    # la, lc - plot label and csv header lable
    # lb - rename parameter array to add in plot and header later
    # ti - plot title
    # meta1 - csv metadata
    # ch - check value, check for only one input parameter array, or none for one contour
    if alpha != 0:
        alpha = np.array(alpha)
        k = len(alpha)
        parj = np.zeros([k,6])
        la = r'$\alpha$'; lc = 'alpha'
        lb = np.copy(alpha)
        ti = r'$\beta_L$=%s, $\eta$=%s, $\rho$=%s, $\i$=%s' % (str(round(par[1],3)),str(round(par[2],3)),str(round(par[3],3)),str(round(par[4],3)))
        meta1 = ['# beta_L=%s'%par[1],'eta=%s'%par[2],'rho=%s'%par[3],'i=%s'%par[4]]
        # add input array values to iteration parameters as appropriate
        for j in range(k):
            parj[j,:] = np.array([alpha[j],par[1],par[2],par[3],par[4],0.])
        ch = ch + 1
    if beta_L != 0:
        beta_L = np.array(beta_L)
        k = len(beta_L)
        parj = np.zeros([k,6])
        la = r'$\beta_L$'; lc = 'beta_L'
        lb = np.copy(beta_L)
        ti = r'$\alpha$=%s, $\eta$=%s, $\rho$=%s, $\i$=%s' % (str(round(par[0],3)),str(round(par[2],3)),str(round(par[3],3)),str(round(par[4],3)))
        meta1 = ['# alpha=%s'%par[0],'eta=%s'%par[2],'rho=%s'%par[3],'i=%s'%par[4]]
        for j in range(k):
            parj[j,:] = np.array([par[0],beta_L[j],par[2],par[3],par[4],0.])
        ch = ch + 1
    if eta != 0:
        eta = np.array(eta)
        k = len(eta)
        parj = np.zeros([k,6])
        la = r'$\eta$'; lc = 'eta'
        lb = np.copy(eta)
        ti = r'$\alpha$=%s, $\beta_L$=%s, $\rho$=%s, $\i$=%s' % (str(round(par[0],3)),str(round(par[1],3)),str(round(par[3],3)),str(round(par[4],3)))
        meta1 = ['# alpha=%s'%par[0],'beta_L=%s'%par[1],'rho=%s'%par[3],'i=%s'%par[4]]
        for j in range(k):
            parj[j,:] = np.array([par[0],par[1],eta[j],par[3],par[4],0.])
        ch = ch + 1
    if rho != 0:
        rho = np.array(rho)
        k = len(rho)
        parj = np.zeros([k,6])
        la = r'$\rho$'; lc = 'rho'
        lb = np.copy(phia)
        ti = r'$\alpha$=%s, $\beta_L$=%s, $\eta$=%s, $\i$=%s' % (str(round(par[0],3)),str(round(par[1],3)),str(round(par[2],3)),str(round(par[4],3)))
        meta1 = ['# alpha=%s'%par[0],'beta_L=%s'%par[1],'eta=%s'%par[2],'i=%s'%par[4]]
        for j in range(k):
            parj[j,:] = np.array([par[0],par[1],par[2],rho[j],par[4],0.])
        ch = ch + 1
    if i != 0:
        i = np.array(i)
        k = len(phia)
        parj = np.zeros([k,6])
        la = r'$i$'; lc = 'i'
        lb = np.copy(i)
        ti = r'$\alpha$=%s, $\beta_L$=%s, $\eta$=%s, $\rho$=%s' % (str(round(par[0],3)),str(round(par[1],3)),str(round(par[2],3)),str(round(par[3],3)))
        meta1 = ['# alpha=%s'%par[0],'beta_L=%s'%par[1],'eta=%s'%par[2],'rho=%s'%par[3]]
        for j in range(k):
            parj[j,:] = np.array([par[0],par[1],par[2],par[3],i[j],0.])
        ch = ch + 1
    # if check value is more than one, too many input parameter arrays given
    if ch > 1:
        return('Please supply at most one parameter to sweep')
    # if check value zero, assume plotting only one contour
    if ch == 0:
        parj = np.zeros([2,6])
        parj[0,:] = par
        parj[1,:] = par
        ti = r'$\alpha$=%s, $\beta_L$=%s, $\eta$=%s, $\rho$=%s, $i$=%s' % (str(round(par[0],3)),str(round(par[1],3)),str(round(par[2],3)),str(round(par[3],3)),str(round(par[4],3)))
        meta1 = ['# alpha=%s'%par[0],'beta_L=%s'%par[1],'eta=%s'%par[2],'rho=%s'%par[3],'i=%s'%par[4]]
    # build sim output array of appropriate size
    # needs as many rows as contours determined by input parameter array
    if k > 1:
        V = np.zeros([k,len(phia)])
    else:
        V = np.zeros([2,len(phia)])
        
    # cp - check progress, total outputs in V
    cp = k*len(phia)
    
    # loop over k rows and len(i) colums of V
    # fill V with average voltage from time series for given params
    # parjj - parameter array for this time series
    # S - state array output from sim
    for j in range(k):
        parjj = parj[j,:]
        for m in range(len(phia)):
            parjj[5] = phia[m]
            S = qSQUID(nStep,tau,s,parjj)
            V[j,m] = sum(S[6,md:])/len(S[6,md:])
            # new progress bar current iter/total iters
            update_progress((m + j*len(phia))/cp)
    # fill out progress bar
    update_progress(1)
    
    # build output for csv
    # join i values and average Voltage matrix
    Sf = np.concatenate((np.matrix(phia),V),axis=0)
    # flip independent axis, i, from horizontal to vertical
    Sf = np.matrix.transpose(Sf)
    # convert from matrix to array to ease csv output
    Sf = np.array(Sf)
    
    # make a figure
    # header - csv header info, param input value for contour
    fig,ax = plt.subplots()
    # one contour, or
    if k == 1:
        ax.plot(phia,V[0])
        header = ['phia','V']
    # k contours
    else:
        header = ['phia']*(k+1)
        for j in range(k):
            ax.plot(phia,V[j],label= la + '=%s' % str(round(lb[j],3)))
            header[j+1] = lc + '=%s' % str(round(lb[j],3))
    ax.set(title=ti,
       xlabel=r'Applied flux, $\phi_a$',
       ylabel=r"Average voltage, $\bar{v}$")
    ax.legend()
    fig.tight_layout()
    
    # build rest of metadata needed for csv
    meta2 = ['# nStep=%s'%nStep,'tau=%s'%tau]
    csvtime = datetime.datetime.now()
    timestr = [datetime.datetime.strftime(csvtime, '# %Y/%m/%d, %H:%M:%S')]
    timeti = str(datetime.datetime.strftime(csvtime, '%Y%m%d%H%M%S'))
    csvtitle='VPhi'+timeti+'.csv'
    pngtitle='VPhi'+timeti+'.png'
    
    # create, write, and save(close) csv
    with open(csvtitle, 'w') as csvFile:
        filewr = csv.writer(csvFile,delimiter=',')
        filewr.writerow(timestr)
        filewr.writerow(meta1)
        filewr.writerow(meta2)
        filewr.writerow(header)
        filewr.writerows(Sf)
    csvFile.close()
    # save figure
    fig.savefig(pngtitle)
    print('csv file written out:', csvtitle)
    print('png file written out:', pngtitle)


# In[18]:


def transfer_fn(nStep,tau,s,par,i,phia):
    """Returns average voltage surface plot and csv
        and transfer function surface plot and csv
    DEPENDENCIES
        numpy as np
        scipy.interpolate
        qSQUID()
    INPUTS
        nStep - number of steps needed in timeseries
        tau - step size for time series
        s - initial state vector
            array[delta_1,delta_2]
        par - parameter vector
            array[alpha,betaL,eta,rho,i,phia]
            alpha - resistance symmetry
            betaL - inductance constant
            eta - inductance symemetry
            rho - resistance symmetry
            i - bias current
            phia - applied mag flux
    OUTPUTS
        average voltage surface plot AveVsurf'datetime'.png
        average voltage surface csv AveVsurf'datetime'.csv
        transfer function surface plot TransferFn'datetime'.png
        transfer function surface csv TransferFn'datetime'.csv"""
    m = len(i)
    n = len(phia)
    l = int(nStep*.1)
    N = m*n
    vp = np.zeros([n,m])
    iv = np.zeros([m,n])
    
    # calculate average voltage surface
    for j in range(0,m):
        for k in range(0,n):
            par[4] = i[j]
            par[5] = phia[k]
            X = qSQUID(nStep,tau,s,par)
            v = np.average(X[6,l:])
            vp[k,j] = v
            iv[j,k] = v
            update_progress((j*n+k)/(m*n))
    update_progress(1)
    
    ## smooth and interpolate over a grid lx dense ##
    l = 1
    inew = np.copy(i)#inew = np.arange(1,2.55,0.05/l)#inew = np.arange(0.8,3.1,0.1/l)
    phianew = np.copy(phia)#phianew = np.arange(0.,.5,.03125/l)#phianew = np.arange(0.,.55,.025/l)

    x, y = np.meshgrid(phia,i)
    xnew, ynew = np.meshgrid(phianew,inew)
    z = np.copy(iv)
    tck = interpolate.bisplrep(y, x, iv, s=.05) # s = smoothing
    ivi = interpolate.bisplev(ynew[:,0], xnew[0,:], tck)
    
    # find gradient of surface
    dv = np.gradient(ivi,inew,phianew)
    
    # filename stuff
    # build rest of metadata needed for csv
    meta1 = ['# alpha=%s, betaL=%s, eta=%s, rho=%s' %(par[0],par[1],par[2],par[3])]
    meta2 = ['# nStep=%s'%nStep,'tau=%s'%tau]
    meta3 = ['# values shown are vbar, representing a surface in dimensions i (vertical) and phia (horizontal)']
    csvtime = datetime.datetime.now()
    timestr = [datetime.datetime.strftime(csvtime, '# %Y/%m/%d, %H:%M:%S')]
    timeti = str(datetime.datetime.strftime(csvtime, '%Y%m%d%H%M%S'))
    csvtitle='AveVsurface'+timeti+'.csv'
    pngtitle='AveVsurface'+timeti+'.png'
    
    # create, write, and save(close) average voltage surface csv
    with open(csvtitle, 'w') as csvFile:
        filewr = csv.writer(csvFile,delimiter=',')
        filewr.writerow(timestr)
        filewr.writerow(meta1)
        filewr.writerow(meta2)
        filewr.writerow(meta3)
        filewr.writerows(iv)
    csvFile.close()
    print('csv file written out:', csvtitle)
    
    
    # plot average voltage surface
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.copy(i)
    y = np.copy(phia)
    x, y = np.meshgrid(y, x)
    z = np.copy(iv)
    ax.plot_wireframe(x, y, z) # cmap='terrain'
    # note the xlabel and ylabel are reversed, this is correct
    ax.set(ylabel=r'bias current $i$',
           xlabel=r'applied flux $\phi_a$',
           zlabel=r'average voltage $\bar{v}$',
          title = r'$\bar{v}(i,\phi_a)$; $\alpha$=%s, $\beta_L$=%s, $\eta$=%s,$\rho$=%s' %(par[0],par[1],par[2],par[3]))
    fig.tight_layout()
    fig.savefig(pngtitle)
    print('png file written out:', pngtitle)
    
    # modify file stuff to ouput transfer function surface
    meta3 = ['# values shown are dvbar/dphia, the transfer function in dimensions i (vertical) and phia (horizontal)']
    csvtitle='TransferFn'+timeti+'.csv'
    pngtitle='TransferFn'+timeti+'.png'
    
    # create, write, and save(close) transger function csv
    with open(csvtitle, 'w') as csvFile:
        filewr = csv.writer(csvFile,delimiter=',')
        filewr.writerow(timestr)
        filewr.writerow(meta1)
        filewr.writerow(meta2)
        filewr.writerow(meta3)
        filewr.writerows(dv[1])
    csvFile.close()
    print('csv file written out:', csvtitle)
    
    
    # plot transfer function
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x = np.copy(inew)
    y = np.copy(phianew)
    x, y = np.meshgrid(y, x)
    z = np.copy(dv[1]) # partial dvbar/dphia
    ax.plot_wireframe(x, y, z) # cmap='terrain'
    # note the xlabel and ylabel are reversed, this is correct
    ax.set(ylabel=r'bias current $i$',
           xlabel=r'applied flux $\phi_a$',
           zlabel=r'transfer function $\partial\bar{v}/\partial\phi_a$',
          title = r'$\bar{v}(i,\phi_a)$; $\alpha$=%s, $\beta_L$=%s, $\eta$=%s,$\rho$=%s' %(par[0],par[1],par[2],par[3]))
    ax.view_init(65,-60)
    fig.tight_layout()
    fig.savefig(pngtitle)
    print('png file written out:', pngtitle)
    return(iv)

