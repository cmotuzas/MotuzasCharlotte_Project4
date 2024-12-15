# Project 4 

import numpy as np 
import matplotlib.pyplot as plt 

# develop a python function that solves the one dimensional, time-dependent Schroedinger Eqn
# section 9.2 of the textbook 
# equations 9.32 and 9.40 in the text 
# main function can call additional functions that you write 

# Taken from Lab 10 

def make_tridiagonal(N,b,d,a): 
    '''function that returns a tridiagonal matrix given the matrix size (N), and three values d, b, and a. The returned matrix will be N by N, and the order of the tridiagonal will be [a,d,b]'''
    A = d*np.eye(N)+a*np.diagflat(np.ones(N-1),1)+b*np.diagflat(np.ones(N-1),-1)
    return A

def spectral_radius(A): 
    '''Function that computes the eigenvalues of an input 2D array A, and returns the eigenvalue with the maximum absolute value.'''
    eigenvalues = np.linalg.eig(A)[0]
    max_eigenvalue = np.max(np.abs(eigenvalues))
    return max_eigenvalue

# New Function

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
    h = length/(nspace-1) # grid size 
    coeff = (hbar**2)/(2*m*(h**2))
    sigma0 = wparam[0]
    x0 = wparam[1]
    k0 = wparam[2]
    x = np.linspace(-length/2, length/2, nspace)  # Spatial grid
    t = np.linspace(0, ntime * tau, ntime)  # Time grid
    V = np.zeros(nspace)
    for i in potential: 
        V[i] = 1

    # Initialize solution array and initial conditions
    psi = np.zeros((nspace, ntime),dtype=complex)
    psi[:, 0] = (1/(np.sqrt(sigma0*np.sqrt(np.pi))))*np.exp(1j*k0*x)*np.exp(-((x-x0)**2)/(2*sigma0**2))      # Initial condition
    #print(np.real(psi[:, 0]))

    d = -2
    b = 1
    a = 1
    H = make_tridiagonal(nspace, b, d, a)
    H[0,0] = d
    H[0, -1] = a
    H[-1, 0] = b
    H[-1,-1] = d
    H[-1,-2] = a 
    H[0,1] = b

    H = -coeff*H + V*np.identity(nspace)

    if method == 'ftcs': 
        A = (np.identity(nspace) - (1j*tau/hbar)*H)

        # Check spectral radius
        sr = spectral_radius(A)
        print(sr)
        if sr - 1 > 1e-5:
            raise ValueError("Unstable FTCS method: spectral radius > 1.")


    elif method == 'crank': 
        A = np.linalg.inv((np.identity(nspace) + (1j*tau/2*hbar)*H)).dot(np.identity(nspace) - (1j*tau/2*hbar)*H)

    else: 
        raise ValueError("Invalid method. Choose 'ftcs' or 'crank'.")
    
    # compute solution
    
    total_prob = np.zeros(ntime)
    prob = np.empty([nspace,ntime])
    for istep in range(1,ntime):
        psi[:, istep] = A.dot(psi[:, istep-1])
        prob[:,istep] = np.abs(psi[:,istep] * np.conjugate(psi[:,istep]))    
        total_prob[istep] = np.sum(np.abs(psi[:, istep])**2) * h  # Normalize probability
    
    return psi, x, t, prob

# see eqn 9.42 in the text as well 
# periodic boundary conditions, see lab 11 
# return 2D array, two 1D arrays giving x and t, and 1D array that gives total prob for each timestep (conserved)
# additional sch_plot function that uses output to visualize results 
# plot of psi at a specific time t
# plot of prob, plot of the particle prob density at a specific time 
# numpy.conjugate to do complex conjugation 

def sch_plot(psi,x,t,prob,ntime,plot,save): 
    if plot == 'psi':
       # from lab 11 

        plotskip = round(0.05*ntime)
        fig, ax = plt.subplots()
        # space out the plots vertically to make the visualization clearer
        yoffset = psi[:,0].max() - psi[:,0].min()
        for i in np.arange(len(t)-1,0,-plotskip): 
            ax.plot(x, psi[:,i]+yoffset*i/plotskip,label = 't ={:.3f}'.format(t[i]))
        ax.set_xlabel('X position')
        ax.set_ylabel('$\\psi$(x,t) [offset]')

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax.set_title('Wave Propagation with Offsets')

        if save == 1: 
            plt.savefig('MotuzasCharlotte_Fig_{}.png'.format(plot))

        plt.show()
    elif plot == 'prob':
        plotskip = round(0.05*ntime)
        fig, ax = plt.subplots()
        # space out the plots vertically to make the visualization clearer
        yoffset = prob[:,1].max() - prob[:,1].min()
        for i in np.arange(len(t)-1,0,-plotskip): 
            ax.plot(x, prob[:,i]+yoffset*i/plotskip,label = 't ={:.3f}'.format(t[i]))
        ax.set_xlabel('X position')
        ax.set_ylabel('P(x,t) [offset]')

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        ax.set_title('Probability Distribution with Offsets')

        if save == 1: 
            plt.savefig('MotuzasCharlotte_Fig_{}.png'.format(plot))

        plt.show()
    else: 
        raise ValueError("Invalid plot type. Choose 'psi' or 'prob'.")

    
    return 

ntime = 2000
psi, x, t, prob = sch_eqn(200, ntime, 0.03, method='crank', length=200, potential = [], wparam = [10, 0, 0.5])
sch_plot(psi,x,t,prob,ntime,'prob',1)

