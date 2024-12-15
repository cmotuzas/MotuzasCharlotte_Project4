# Project 4 

import numpy as np 
import scipy as scp 
import matplotlib.pyplot as plt 

# develop a python function that solves the one dimensional, time-dependent Schroedinger Eqn
# section 9.2 of the textbook 
# equations 9.32 and 9.40 in the text 
# main function can call additional functions that you write 

def sch_eqn(nspace, ntime, tau, method='ftcs', length=200, potential = [], wparam = [10, 0, 0.5]):
    '''
    method: string, either ftcs or crank
    length: float, size of spatial grid. Default to 200 (grid extends from -100 to +100)
    potential: 1-D array giving the spatial index values at which the potential V(x) should be set to 1. Default to empty. For example, [25, 50] will set V[25] = V[50] = 1.
    wparam: list of parameters for initial condition [sigma0, x0, k0]. Default [10, 0, 0.5].
    
    returns a 2d array returning psy as a function of x and t, and the corresponding 2D arrays x and t as grid values
    in addition, returns the total probability computed for each time step. '''
    
    m = 0.5
    hbar = 1
    coeff = (hbar**2)/(2*m)

    if method == 'ftcs': 
        print('ftcs')

    elif method == 'crank': 
        print('crank')
        
    else: 
        print("Please enter either 'ftcs' or 'crank' as the method input")    
    
    
    
    
    
    
    
    return print('Not Done Yet')

# see eqn 9.42 in the text as well 
# periodic boundary conditions, see lab 11 
# return 2D array, two 1D arrays giving x and t, and 1D array that gives total prob for each timestep (conserved)
# additional sch_plot function that uses output to visualize results 
# plot of psi at a specific time t
# plot of prob, plot of the particle prob density at a specific time 
# numpy.conjugate to do complex conjugation 

# use comments to describe the origins of the code, if taken from NM4P or prior labs 

# report, function documentation, report on how you tested your functions 
# write so that it may be used by an end user 