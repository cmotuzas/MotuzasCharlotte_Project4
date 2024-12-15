# Project 4 

import numpy as np 
import scipy as scp 
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

def make_initialcond(sigma_0,k_0,x): 
    '''This function returns a vector a_0 of initial conditions specified by the wavelet a(x,0) = exp(-x^2/(2sigma_0^2))*cos(k_0*x), given constants sigma_0, k_0, and the spatial grid x (given as a vector)'''
    a_0 = np.exp((-x**2)/(2*sigma_0**2))*np.cos(k_0*x)
    return a_0

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
    V = np.zeros(len(x))
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
    H[0, -1] = b
    H[-1, 0] = a
    H = -coeff*H + V*np.identity(nspace)

    if method == 'ftcs': 
        A = (np.identity(nspace) - (1j*tau/hbar)*H)

        # Check spectral radius
        sr = spectral_radius(A)
        print(sr)
        if sr - 1 > 1e-5:
            print("Warning: Unstable integration. Spectral radius > 1.")
        else: 
            print("Seems good!!")

    elif method == 'crank': 
        A = np.linalg.inv((np.identity(nspace) - (1j*tau/hbar)*H)).dot(np.identity(nspace) - (1j*tau/hbar)*H)

    else: 
        print("Please enter either 'ftcs' or 'crank' as the method input")    
    
        # compute solution
    
    prob = np.empty([nspace,ntime])
    for istep in range(1,ntime):
        psi[:, istep] = A.dot(psi[:, istep-1])
        prob[:,istep] = np.abs(psi[:,istep] * np.conjugate(psi[:,istep]))    
    
    return psi, x, t, prob

# see eqn 9.42 in the text as well 
# periodic boundary conditions, see lab 11 
# return 2D array, two 1D arrays giving x and t, and 1D array that gives total prob for each timestep (conserved)
# additional sch_plot function that uses output to visualize results 
# plot of psi at a specific time t
# plot of prob, plot of the particle prob density at a specific time 
# numpy.conjugate to do complex conjugation 


def sch_plot(psi,x,t,prob): 
    fig_psi = plt.figure()
    for i in range(8): 
        plt.plot(x,np.real(psi[:,50*i]),label='{}'.format(50*i))
    plt.title('Schrodinger Wave Equation Results')
    plt.xlabel('Position (x)')
    plt.ylabel('$\\psi$ (x, t)')
    plt.legend()
    plt.show()

    fig2 = plt.figure()
    for i in range(1,8): 
        plt.plot(x,prob[:,50*i],label='{}'.format(50*i))
    plt.title('Particle Probability Density')
    plt.xlabel('Position (x)')
    plt.ylabel('$|\\psi|^2$ (x, t)')
    plt.legend()
    plt.show()

    return 


psi, x, t, prob = sch_eqn(100, 500, 0.03, method='crank', length=200, potential = [], wparam = [10, 0, 0.5])
sch_plot(psi,x,t,prob)