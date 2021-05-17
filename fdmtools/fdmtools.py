# Dependencies
import warnings

import h5py as h5
import numpy as np
import scipy as sp
import scipy.optimize as scopt
import scipy.interpolate as scinterp
import scipy.integrate as scinteg
import scipy.special as scsp
import scipy.stats as scstat
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
plt.rc('font', size=18)

import pyshtools as pysh


# __all__ = ['rvir_calc']


# Constants
G = 4.30091727e-6               # Gravitational constant in kpc / M_sun * (km / s)^2

# Cosmology
Om = 0.2865                     # Omega matter
rho_crit = 133.36363195167573   # Critical background density (solMass / kpc^3)
rho_m = rho_crit * Om           # Background matter density at present
overdensity = 347.0             # Overdensity for spherical collapse
rho_vir = overdensity * rho_m   # Average density within the virial radius

# Quantum Mechanics
m_s = 8.1e-23                   # FDM particle mass in eV / c^2
m22 = 0.81                      # FDM particle mass in eV / c^2 * 10^-22
m23 = 8.1                       # FDM particle mass in eV / c^2 * 10^-23
m2h2 = 0.0017850762909265429    # Scaled unit (m_s / h_bar)^2 for Schrodinger Equation in astrophysical units: (kpc * km / s)^-2 



def t_freefall(r, M_enc):
    """
    Compute the freefall time at a given radius
    
    Parameters
    ----------
    M_enc : numeric
        Mass enclosed within the radius in solar masses
    r : numeric
        Radius in kpc
    
    Returns
    -------
    T_ff : numeric
        Freefall time in Gyr
    """
    G_Gyr = G * 1.023**2.
    T_ff = np.sqrt((np.pi**2. * r**3.) / (8. * M_enc * G_Gyr))
    return T_ff


def rvir_calc(M_halo):
    """
    Compute the virial radius of a halo
    
    Parameters
    ----------
    M_halo : numeric
        Halo mass in solar masses
    
    Returns
    -------
    r_vir : numeric
        Virial radius in kpc
    """
    r_vir = (M_halo / (4.0 / 3.0 * np.pi * rho_vir))**(1.0 / 3.0)
    return r_vir


def r_core(M_halo, powerlaw=1/3):
    """
    Compute the core radius of a FDM halo given the Schive et al. (2014) Core-Halo relation
    
    Parameters
    ----------
    M_halo : numeric
        Halo mass in solar masses
    
    Returns
    -------
    r_core : numeric
        Core radius in kpc
    """
    r_c = 1.6 / m22 * (M_halo / 1e9)**(-powerlaw)
    return r_c


def rho_core(r, r_c):
    """
    Compute the core density profile of a FDM halo given the Schive et al. (2014) Core-Halo relation
    
    Parameters
    ----------
    r : numpy array
        Array of radial distances
    M_halo : numeric
        Halo mass in solar masses
    r_c : numeric
        Core radius of the FDM halo in kpc
    
    Returns
    -------
    rho_c : numeric
        Density at radius r in solar masses per kpc^3
    """
    rho_c = (1.9 * m23**(-2.) * r_c**(-4.)) / (1. + 0.091 * (r / r_c)**2.)**8.
    return rho_c * 1e9


def phi_calc(r, rho):
    """
    Compute the gravitational potential at a radius given the density, using Poisson's equation
    
    Parameters
    ----------
    r : numpy array
        Array of radial distances
    rho : lambda function or spline object
        Lambda function or cubic spline object that returns the value of the density as a function of radius
    
    Returns
    -------
    phi : numpy array
        Values of the gravitational potential corresponding to the input radii in (km / s)^2

    """
    integrand1 = lambda r: rho(r) * r**2
    integrand2 = lambda r: rho(r) * r

    warnings.filterwarnings("ignore")

    integral1, err = scinteg.quad(integrand1, 0., r)
    integral2, err = scinteg.quad(integrand2, r, np.inf)

    return -4. * np.pi * G * (1. / r * integral1 + integral2)


def initiate_potential(rho_in, core=False, M_halo=None, powerlaw=1/3):
    """
    Initiate functions for calculating the potential and density as a function of radius
    
    Parameters
    ----------
    rho_in : function
        A Python function that returns the density in Solar Masses per kpc^3 at a given radius (in kpc)
    core : bool (optional, default: False)
        Superpose a soliton core based on a powerlaw Core-Halo relation
    M_halo : numeric
        Halo mass in solar masses for determining the core density; required only if core==True
    powerlaw : numeric (default: 1/3)
        Powerlaw relation between the halo mass and the core mass. Default is 1/3, following the Schive et al. (2014) relation
    
    Returns
    -------
    phi_out : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the potential in (km / s)^2 as a function of radius (in kpc)
    rho_out : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the density in Solar Masses per kpc^3 as a function of radius

    """
    rspline = np.logspace(-6, 6, 1024)
    rho_halo = rho_in(rspline)

    if core==False:

        rho_out = scinterp.CubicSpline(rspline, rho_halo)

        phispline = np.array([phi_calc(r, rho_in) for r in rspline])
        phi_out = scinterp.CubicSpline(rspline, phispline)

    elif core==True:

        if M_halo is None:
            print('Must input M_halo parameter for core-halo mass relation if core==True')

        r_c = r_core(M_halo, powerlaw=powerlaw)
        rho_c = rho_core(rspline, r_c)

        if np.any(rho_c > rho_halo)==False:
            print('Core density never reaches halo density; please input larger mass halo or different powerlaw.')

        transition = np.max(np.where(rho_c[rho_halo>rho_m] > rho_halo[rho_halo>rho_m])) + 1

        rho_combined = np.concatenate((rho_c[:transition], rho_halo[transition:]))

        rho_out = scinterp.CubicSpline(rspline, rho_combined)
        phispline = np.array([phi_calc(r, rho_out) for r in rspline])
        phi_out = scinterp.CubicSpline(rspline, phispline)

    dM = lambda r: 4. * np.pi * rho_out(r) * r**2
    M_total, err = scinteg.quad(dM, 0., np.inf)

    return phi_out, rho_out, M_total


def DF_invert(phi, rho, M):
    """
    Inversion formula to calculate the distribution function f(E) of a potential given an array of total energy

    Parameters
    ----------
    phi : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the potential in (km / s)^2 as a function of radius (in kpc)
    rho : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the density in Solar Masses per kpc^3 as a function of radius
    M : numeric
        Total mass of the halo in Solar Masses
    
    Returns
    -------
    f_E : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the distribution function as a function of the total energy in (km / s)^2

    """
    Phimin = -phi(0.) * 1e-6
    Phimax = -phi(0.) * (1. - 1e-6)
    Phispline = np.linspace(Phimin, Phimax, 1024)
    rspline = np.array([phi.solve(-Phi, extrapolate=False)[0] for Phi in Phispline])
    nuspline = rho(rspline) / M

    nu = scinterp.CubicSpline(Phispline, nuspline)
    dnu_dPhi = nu.derivative()

    def integrand(Phi_integ, Eps_integ):
        return 1. / np.sqrt(Eps_integ - Phi_integ) * dnu_dPhi(Eps_integ)

    Epsspline = Phispline
    Hspline = -Epsspline

    integralspline = np.array([scinteg.quad(integrand, 0., Eps, args=(Eps,))[0] for Eps in Epsspline])

    integral = scinterp.CubicSpline(Epsspline, integralspline)
    dintegral_dEps = integral.derivative()

    f_Espline = np.array([1. / (np.sqrt(8.) * np.pi**2) * dintegral_dEps(Eps) for Eps in Epsspline])
    f_E = scinterp.CubicSpline(np.flip(Hspline), np.flip(f_Espline))

    return f_E


def g_E(phi):
    """
    Determination of the classical density of states (typically labeled g(E)) given an input potential 

    Parameters
    ----------
    phi : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the potential in (km / s)^2 as a function of radius (in kpc)

    Returns
    -------
    g_E : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the density of states as a function of the total energy in (km / s)^2

    """
    Emin = phi(0.) * (1. - 1e-6)
    Emax = phi(0.) * 1e-6
    Espline = np.linspace(Emin, Emax, 1024)

    def integrand(r, Einteg):
        return r**2 * np.sqrt(2. * (Einteg - phi(r)))

    g_Espline = np.array([(4. * np.pi)**2 * scinteg.quad(integrand, 0., phi.solve(E, extrapolate=False)[0], args=(E,))[0] for E in Espline])

    g_E = scinterp.CubicSpline(Espline, g_Espline)

    return g_E


def dndE(E_nl, phi, bins=100, sparse_adjust=True):
    """
    Numerical calculation of the density of states (or states per energy bin dn / dE) as a function of total energy

    """
    m_shape = np.zeros(np.shape(E_nl)[0], dtype=int)
    for i in range(np.shape(E_nl)[0]):
        m_shape[i] = 2 * i + 1
    E_nl_repeat = np.repeat(E_nl, m_shape, axis=0)
    Eflat = E_nl_repeat.flatten()[E_nl_repeat.flatten()!=0]
    bindata, binedges = np.histogram(Eflat, bins=100, range=(phi(0.), np.max(Eflat)))

    if sparse_adjust==False:
        dnde = np.zeros_like(Eflat)
        for i in range(int(len(Eflat))):
            dnde[i] = bindata[np.argmax(binedges>Eflat[i]) - 1]
        return Eflat, dnde

    elif sparse_adjust==True:
        bindata1 = np.zeros(len(bindata))
        indx=0
        for i in range(len(bindata)):
            if i==0:
                continue
            if bindata[i-1] == 0 and bindata[i] != 0:
                multiplier = i - indx
                indx = i
            else:
                multiplier = 1.
            bindata1[i] = bindata[i] / multiplier

        dnde = np.zeros_like(Eflat)
        for i in range(int(len(Eflat))):
            dnde[i] = bindata1[np.argmax(binedges>Eflat[i]) - 1]
        return Eflat, dnde


def valsvecs(r, l, phi, Emax, nsteps=1024, verbose=False):
    """
    For a given value of l, a gravitational potential, and a maximum total energy, returns the eigenvalues and eigenvectors of the Schrodinger equation.
    Uses the shooting method for l=0 and the finite difference method for l>0.

    """
    r_guess, step = np.linspace(np.min(r), np.max(r), nsteps, retstep=True)
    Dr2_0 = -1.0 / (2.0 * m2h2) / (step**2) * np.ones(nsteps) * (-2.)
    Dr2_1 = -1.0 / (2.0 * m2h2) / (step**2) * np.ones(nsteps - 1)
    Vvec = l * (l + 1) / (2. * r_guess**2. * m2h2) + np.array([phi(ri) for ri in r_guess])
        
    E_n, u_vecs_T = sp.linalg.eigh_tridiagonal(Dr2_0 + Vvec, Dr2_1, select='v', select_range=(-np.inf, Emax))
    u_vecs = u_vecs_T.T
    
    R_n = np.zeros([len(E_n), len(r)])
    
    for i in range(len(E_n)):

        if l==0:

            def fun(r, u, p):
                E = p[0]
                return np.vstack((u[1], u[0] * ((l * (l + 1)) / (r**2.) + 2. * m2h2 * (phi(r) - E))))

            def bc(ya, yb, p):
                return np.array([ya[0], yb[0], ya[1] - 0.1])

            u_i = np.zeros((2, r.size))
            u_i[0] = 1.

            res = scinteg.solve_bvp(fun, bc, r, u_i, p = [E_n[i]], max_nodes = nsteps*2)
            E_n[i] = res.p[0]
            R0 = res.sol(r)[0] / r
            A = 1. / np.sqrt(np.trapz(R0**2. * r**2., r))
            R_n[i] = R0 * A
            R_n[i, 0] = R_n[i, 1]

        else:
            R0 = u_vecs[i] / r
            A = 1. / np.sqrt(np.trapz(R0**2. * r**2., r))
            R_n[i] = R0 * A

    if np.all(np.diff(E_n, n=2)[:-1] < 0)==False:
        print('Warning! here may be missing and/or duplicated l='+str(l)+' eigenvalues; consider increasing nsteps parameter')
        if verbose==True:
            print(E_n)
            print(np.diff(E_n, n=2)[:-1])

    return E_n, R_n



def compute_radial_solutions(phi, Mh0, r_min=None, r_max=None, nsteps=1024, Emax_factor=1, verbose=True):
    """
    Compute all eigenvalues and eigenvectors (for all l) for the Schrodinger equation
    """
    if r_min is None:
        r_min = 0.001
    if r_max is not None:
        Emax = phi(r_max) / 2.
        r_end = phi.solve(Emax, extrapolate=False)[0]
        print('r_end = '+str(r_end)+' kpc (eigenmodes calculated out to this radius, so that a good approximation of the waveform can be calculated at r_max)')
    if r_max is None:
        r_Emax = rvir_calc(Mh0) * Emax_factor
        Emax = phi(r_Emax) / 2.
        r_end = phi.solve(Emax, extrapolate=False)[0]
        print('r_end = '+str(r_end)+' kpc (eigenmodes calculated out to this radius, so that a good approximation of the waveform can be calculated at r_max)')
    rsolve = np.linspace(r_min, r_end, nsteps)

    E_nl_grid = []
    R_nl_grid = []
    l = 0
    shape_n = 1
    while shape_n>0:
        E_nl, R_nl = valsvecs(rsolve, l, phi, Emax, nsteps=nsteps, verbose=verbose)
        shape_n = len(E_nl)
        if shape_n==0:
            break
        if l==0:
            n_max = shape_n
        E_nl_array = np.zeros(n_max)
        R_nl_array = np.zeros([n_max, len(rsolve)])
        E_nl_array[:shape_n] = E_nl
        R_nl_array[:shape_n] = R_nl
        E_nl_grid.append(E_nl_array)
        R_nl_grid.append(R_nl_array)
        if verbose==True:
            print('l = '+str(l)+'; n_max = '+str(shape_n))
        l+=1
    l_max = l

    print('l_max = '+str(l_max)+'; n_max = '+str(n_max))

    return rsolve, np.array(E_nl_grid), np.array(R_nl_grid), (l_max, n_max)



def compute_radial_solutions_from_list(phi, states, r_min=0.001, r_max=100., nsteps=1024, verbose=False):
    """
    Compute all eigenvalues and eigenvectors (for all l) for the Schrodinger equation
    """
    rsolve = np.linspace(r_min, r_max, nsteps)
    Emax = phi(r_max) / 2.

    E_nl_list = np.zeros(len(states))
    R_nl_list = np.zeros((len(states), nsteps))
    llist = np.unique(states[:, 0])
    for l in llist:
        E_nl, R_nl = valsvecs(rsolve, l, phi, Emax, nsteps=nsteps, verbose=verbose)
        shape_n = len(E_nl)
        if len(E_nl)==0:
            print('No eigenmodes were found for l = '+str(l))
            break
        for i in np.where(states[:, 0]==l)[0]:
            n = states[i, 1]
            E_nl_list[i] = np.copy(E_nl[n])
            R_nl_list[i] = np.copy(R_nl[n])
    return rsolve, E_nl_list, R_nl_list


def amp_from_fE(r, E_nl, R_nl, fE, M_target):
    """
    Compute the eigenmode amplitudes a_nlm directly from an input distribution function
    """
    phi_ang = 0.
    theta_ang = 0.
    shape_nl = np.shape(E_nl)
    Y_lm = np.zeros((shape_nl[0], shape_nl[0] * 2 + 1), dtype=complex)

    for l in range(shape_nl[0]):
        for m in range(-l, l+1):
            Y_lm[l, m] = scsp.sph_harm(m, l, phi_ang, theta_ang)

    a_nlm = np.zeros_like(E_nl)

    a_nlm[E_nl!=0.] = np.sqrt(fE(E_nl[E_nl!=0.]))
    a_nlm[np.isnan(a_nlm)] = 0.
    den_total = np.zeros(len(r))

    for l in range(shape_nl[0]):
        for n in range(shape_nl[1]):
            if a_nlm[l, n]==0.:
                continue
            else:
                for m in range(-l, l+1):
                    den_nlm = (a_nlm[l, n] * R_nl[l, n] * np.abs(Y_lm[l, m]))**2.
                    den_total += den_nlm

    M_norm = np.trapz(4. * np.pi * r**2. * den_total, r)
    den_total = den_total / M_norm * M_target
    a_nlm = a_nlm * np.sqrt(M_target / M_norm)
    norm = np.sqrt(M_target / M_norm)

    return a_nlm, den_total, norm


def amp_from_schwarzschild(r, rmax, rho, M_total, E_nl, R_nl, a_init=None, skip=1, max_iterations=2500, verbose=True):
    """
    Compute the eigenmode amplitudes a_nl such that (a_nl @ R_nl)**2 = the density profile (Schwarzschild method).
    Degenerate m modes are not accounted for.
    """
    rfit = r[r<rmax]
    den_target = rho(rfit[::skip])
    Eflat = E_nl.flatten()[E_nl.flatten()!=0]
    Rflat_long = np.reshape(R_nl, (np.shape(R_nl)[0] * np.shape(R_nl)[1], np.shape(R_nl)[2]))[E_nl.flatten()!=0]
    Rflat = Rflat_long[:, :len(rfit):skip]
    
    func_to_min = lambda amps: np.sum(((den_target - (amps**2.) @ (Rflat**2.)) / (den_target + (amps**2.) @ (Rflat**2.)))**2)

    if a_init is None:
        aflat_init = np.ones(np.shape(Eflat)) * np.sqrt(M_total) * 0.0001
    else:
        aflat_init = a_init.flatten()[E_nl.flatten()!=0]

    if verbose==True:
        global Nfeval
        Nfeval = 1
        def callbackF(Xi):
            global Nfeval
            if Nfeval%20==0:
                print('{0:4d}   {1: 3.6f}'.format(Nfeval, func_to_min(Xi)))
            Nfeval += 1
        new_amps = scopt.minimize(func_to_min, aflat_init, callback=callbackF, options={'maxiter': max_iterations})
        aflat = np.abs(new_amps.x)
        print(new_amps.success)
        print('Optimization complete after '+str(new_amps.nit)+' iterations')
        print(new_amps.message)
    
    else:
        new_amps = scopt.minimize(func_to_min, aflat_init, options={'maxiter': max_iterations})
        aflat = np.abs(new_amps.x)
        print('Optimization complete after '+str(new_amps.nit)+' iterations')

    den_total = (aflat**2.) @ (Rflat_long**2.)
    a_nl = np.zeros_like(E_nl)
    a_nl[E_nl != 0] = aflat
    return a_nl, den_total


def amp_rad_to_spher(a_nl):
    """
    Convert a_nl to a_nlm (to account for degenerate m modes and allow for creation of 3D spherical halos)
    """
    dim = np.shape(a_nl)
    a_scale = np.sqrt(4. * np.pi / np.repeat(np.arange(1., 2. * dim[0] + 1., 2.), dim[1]).reshape(dim))
    a_nlm = a_nl * a_scale
    return a_nlm


def amp_spher_to_rad(a_nlm):
    """
    Convert a_nlm to a_nl (to consolidate degenerate m modes and allow for efficient calculation of the 1D density profile)
    """
    dim = np.shape(a_nlm)
    a_scale = np.sqrt(4. * np.pi / np.repeat(np.arange(1., 2. * dim[0] + 1., 2.), dim[1]).reshape(dim))
    a_nl = a_nlm / a_scale
    return a_nl


def random_phase(lmax, nmax):
    """
    Output a random phase for each eigenmode (n, l, and m) given lmax and nmax
    """
    phase_nlm = np.zeros((lmax, nmax, 2 * lmax + 1))
    for l in range(lmax):
        for n in range(nmax):
            for m in range(-l, l + 1):
                phase_nlm[l, n, m] = np.random.rand() * 2. * np.pi
    
    return phase_nlm


def amp_to_clm(aphase_n, RatShell, lmax):
    """
    Convert amplitude and radial eigenfunction into clm spectrum for SHTools
    """
    anlm = aphase_n * RatShell[:, None]
    anlm1 = anlm[:, :lmax+1]
    anlm2 = np.roll(np.flip(anlm[:, lmax:], axis=1), 1)
    power = np.zeros((2, lmax, lmax+1), dtype = complex)
    power[0] = anlm1
    power[1] = anlm2
    
    return power
    

def solve_amps_itr(M, phi_in, rho_in, fE, rbox, rfit=None, converge_target=0.01, maxiter=3, method='Schwarzschild', amps_guess='DF', schwarz_iter=500, nsteps=1024, full_output=False):
    """
    Solve for best fit amplitudes by iterating several times
    """
    r_arr = np.logspace(-3, 3, 100)

    new_M = 4. * np.pi * np.trapz(r_arr**2. * rho_in(r_arr), r_arr)
    print('Initial mass = 10^'+str(np.log10(new_M)))

    rsolve_list = []
    dens_list = []

    rho = rho_in
    phi = phi_in
    
    for i in range(maxiter):
        print('Iteration #'+str(i+1))
        print('Solving for eigenmodes...')
        rsolve, E_nl_grid, R_nl_grid, shape_nl = compute_radial_solutions(phi, M, r_max=rbox, nsteps=nsteps, verbose=False)

        M_target = np.trapz(4. * np.pi * rsolve**2. * rho(rsolve), rsolve)
        a_nlm, den_DF, norm = amp_from_fE(rsolve, E_nl_grid, R_nl_grid, fE, M_target)

        print('Solving for amplitudes...')

        if rfit==None:
            rfit=rbox
        
        if method=='DF':
            dens = den_DF

        elif method=='Schwarzschild':
            if amps_guess=='DF':
                a_nl_compare = amp_spher_to_rad(a_nlm)
                a_nl, den_schwarz = amp_from_schwarzschild(rsolve, rfit, rho, M, E_nl_grid, R_nl_grid, a_init=a_nl_compare, max_iterations=schwarz_iter, verbose=True)
            elif amps_guess=='flat':
                a_nl, den_schwarz = amp_from_schwarzschild(rsolve, rfit, rho, M, E_nl_grid, R_nl_grid, max_iterations=schwarz_iter, verbose=True)
            else: print('Must specify shape of initial amplitude guess (DF or flat)')
            
            a_nlm = amp_rad_to_spher(a_nl)

            dens = den_schwarz
        
        else: print('Must specify method (DF or Schwarzschild)')

        rho_interp = scinterp.interp1d(rsolve, dens, bounds_error=False, fill_value=(dens[0], 0.), assume_sorted=True)
        rho_in = lambda r: rho_interp(r)

        new_M = 4. * np.pi * np.trapz(r_arr**2. * np.array([rho_in(r) for r in r_arr]), r_arr)
        print('Updated mass = 10^'+str(np.log10(new_M)))

        print('Solving for new potential...')
        phi_new, rho_new, M = initiate_potential(rho_in)
        if i < maxiter - 1:
            fE = DF_invert(phi_new, rho_new, M)

        rsolve_list.append(rsolve)
        dens_list.append(dens)

        phi_diff = ((phi_new(rsolve) - phi(rsolve)) / (phi_new(rsolve) + phi(rsolve)))**2.
        converge_crit = 2. / np.max(rsolve) * np.trapz(phi_diff, rsolve)
        print('Convergence Criterion after Iteration #'+str(i+1)+': D = '+str(converge_crit))
        phi = phi_new
        rho = rho_new
        if converge_crit < converge_target and i > 0:
            print('Convergence criterion met!')
            break

    phase_nlm = random_phase(np.shape(E_nl_grid)[0], np.shape(E_nl_grid)[1])

    print('Completed!')

    if full_output==False:

        return E_nl_grid, R_nl_grid, a_nlm, phase_nlm, rsolve, dens

    elif full_output==True:

        return E_nl_grid, R_nl_grid, a_nlm, phase_nlm, rsolve, dens, rsolve_list, dens_list
    

def build_halo(rshells, E_nl, rsolve, R_nl, a_nlm, phase_nlm, lgrid=None, dt=None, steps=None):
    """
    Build a 3D halo using spherical harmonic transformations
    """
    lmax = np.shape(E_nl)[0]
    nmax = np.shape(E_nl)[1]
    
    if lgrid is None:
        if lmax<60:
            lgrid = 80
        else:
            lgrid = lmax + lmax // 3

    if dt is None and steps is None:

        aphase = (np.cos(phase_nlm) + 1j * np.sin(phase_nlm)) * a_nlm[:,:,None]
        
        clm_setup = pysh.SHCoeffs.from_zeros(lmax=lgrid, kind='complex')
        grid_setup = clm_setup.expand(lmax=lgrid)
        
        grid3d_zeros = np.zeros((len(rshells),) + np.shape(grid_setup.data))
        Rs = np.copy(grid3d_zeros) + rshells[:, None, None]
        thetas = np.copy(grid3d_zeros) + ((-grid_setup.lats() + 90.) / 180. * np.pi)[None, :, None]
        phis = np.copy(grid3d_zeros) + ((grid_setup.lons() - 180) / 180. * np.pi)[None, None, :]
        grid3d = np.zeros((len(rshells),) + np.shape(grid_setup.data), dtype='complex')
        
        RatShell = np.zeros(lmax)
            
        for i in range(len(rshells)):
            for n in range(nmax):
                for l in range(lmax):
                    RatShell[l] = np.interp(rshells[i], rsolve, R_nl[l, n, :len(rsolve)])
                
                power = amp_to_clm(aphase[:, n, :], RatShell, lmax)
                
                clm = pysh.SHCoeffs.from_array(power, normalization='ortho')
                grid = clm.expand(lmax=lgrid)
                
                grid3d[i] += grid.data

    elif dt is None and steps is not None:
        print('If steps parameter is specified, must also specify dt parameter.')

    elif dt is not None and steps is None:
        print('If dt parameter is specified, must also specify steps parameter.')

    elif dt is not None and steps is not None:
        print('Creating halo and evolving for '+str(dt * steps)+' Gyr')

        const = 5.33453e20 * m_s                    # (unit.eV / hbar).to(1 / unit.Gyr) / (c.to(unit.km / unit.s))**2 * m23

        grid3d = []

        for step in range(steps):

            print()

            aphase = (np.cos(phase_nlm) + 1j * np.sin(phase_nlm)) * a_nlm[:,:,None] * np.exp(-1j * E_nl[:,:,None] * const * step * dt)
            
            clm_setup = pysh.SHCoeffs.from_zeros(lmax=lgrid, kind='complex')
            grid_setup = clm_setup.expand(lmax=lgrid)
            
            grid3d_zeros = np.zeros((len(rshells),) + np.shape(grid_setup.data))
            Rs = np.copy(grid3d_zeros) + rshells[:, None, None]
            thetas = np.copy(grid3d_zeros) + ((-grid_setup.lats() + 90.) / 180. * np.pi)[None, :, None]
            phis = np.copy(grid3d_zeros) + ((grid_setup.lons() - 180) / 180. * np.pi)[None, None, :]
            grid3d_temp = np.zeros((len(rshells),) + np.shape(grid_setup.data), dtype='complex')
            
            RatShell = np.zeros(lmax)
                
            for i in range(len(rshells)):
                for n in range(nmax):
                    for l in range(lmax):
                        RatShell[l] = np.interp(rshells[i], rsolve, R_nl[l, n, :len(rsolve)])
                    
                    power = amp_to_clm(aphase[:, n, :], RatShell, lmax)
                    
                    clm = pysh.SHCoeffs.from_array(power, normalization='ortho')
                    grid = clm.expand(lmax=lgrid)
                    
                    grid3d_temp[i] += grid.data

            grid3d.append(grid3d_temp)
    
    return grid3d, Rs, thetas, phis


def plot_slices(grid3d, Rs, thetas, phis, res, rmax, rshells):
    """
    Plot xy, xz, and yz slices for a fdm halo grid
    """
    Rs_flat = Rs.flatten()
    phis_flat = phis.flatten()
    thetas_flat = thetas.flatten()

    xs_flat = Rs_flat * np.sin(thetas_flat) * np.cos(phis_flat)
    ys_flat = Rs_flat * np.sin(thetas_flat) * np.sin(phis_flat)
    zs_flat = Rs_flat * np.cos(thetas_flat)

    dens_flat = np.abs(grid3d.flatten())**2.

    boxedge = np.sqrt(rmax**2 / 3.)

    axis1 = np.linspace(-boxedge, boxedge, res)
    axis2 = np.linspace(-boxedge, boxedge, res)
    Xxy,Yxy,Zxy = np.meshgrid(axis1, axis2, 0., indexing='ij')
    Xxz,Yxz,Zxz = np.meshgrid(axis1, 0. ,axis2, indexing='ij')
    Xyz,Yyz,Zyz = np.meshgrid(0., axis1, axis2, indexing='ij')

    dens_xy = sp.interpolate.griddata((xs_flat, ys_flat, zs_flat), dens_flat, (Xxy, Yxy, Zxy), method='nearest', fill_value = 0.)[:, :, 0]
    dens_xz = sp.interpolate.griddata((xs_flat, ys_flat, zs_flat), dens_flat, (Xxz, Yxz, Zxz), method='nearest', fill_value = 0.)[:, 0, :]
    dens_yz = sp.interpolate.griddata((xs_flat, ys_flat, zs_flat), dens_flat, (Xyz, Yyz, Zyz), method='nearest', fill_value = 0.)[0, :, :]
    
    rshell_dens = np.mean(np.abs(grid3d)**2., axis=(1, 2))
    bar_max = np.log10(np.max(dens_xy))
    bar_min = np.log10(np.min(rshell_dens[rshells < boxedge]) / 10.)
    
    fig, f_axes = plt.subplots(ncols=3, figsize=(24, 8), constrained_layout=True, sharex=False, sharey=False)

    f_axes[0].imshow(np.log10(dens_xy).T,vmin=bar_min,vmax=bar_max,aspect='equal',cmap='PuOr_r',extent=[-boxedge,boxedge,-boxedge,boxedge], origin='lower')
    f_axes[1].imshow(np.log10(dens_xz).T,vmin=bar_min,vmax=bar_max,aspect='equal',cmap='PuOr_r',extent=[-boxedge,boxedge,-boxedge,boxedge], origin='lower')
    ax = f_axes[2]
    pcm = f_axes[2].imshow(np.log10(dens_yz).T,vmin=bar_min,vmax=bar_max,aspect='equal',cmap='PuOr_r',extent=[-boxedge,boxedge,-boxedge,boxedge], origin='lower')

    f_axes[0].set_xlabel('$x$ [kpc]')
    f_axes[1].set_xlabel('$x$ [kpc]')
    f_axes[2].set_xlabel('$y$ [kpc]')
    f_axes[0].set_ylabel('$y$ [kpc]')
    f_axes[1].set_ylabel('$z$ [kpc]')
    f_axes[2].set_ylabel('$z$ [kpc]')
    
    fig.colorbar(pcm, ax=ax, label=r'$\rho$ [$M_\odot$ / kpc$^3$]')
    
    plt.show()
    
    return dens_xy, dens_xz, dens_yz


def halo_to_cube(grid3d, Rs, thetas, phis, res, rmax):
    """
    Project halo to a cube in cartesian coordinates
    """
    Rs_flat = Rs.flatten()
    phis_flat = phis.flatten()
    thetas_flat = thetas.flatten()

    xs_flat = Rs_flat * np.sin(thetas_flat) * np.cos(phis_flat)
    ys_flat = Rs_flat * np.sin(thetas_flat) * np.sin(phis_flat)

    boxedge = np.sqrt(rmax**2 / 3.)

    zs_flat = Rs_flat * np.cos(thetas_flat)

    grid_flat = grid3d.flatten()
    
    x = np.linspace(-boxedge, boxedge, res)
    y = np.linspace(-boxedge, boxedge, res)
    z = np.linspace(-boxedge, boxedge, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    psi = sp.interpolate.griddata((xs_flat, ys_flat, zs_flat), grid_flat, (X, Y, Z), method='nearest', fill_value = 0.)
    
    return psi


def cube_to_ezno_input(psi, res, boxedge, enzo_dens=1.8788e-29, enzo_time=2.519445e17, absorption_boundary=False, zero_momentum=False):

    dens_gal_to_cgs = 6.7679e-32
    dens_to_enzo = dens_gal_to_cgs / enzo_dens
    kpc_to_cm = 3.0857e21

    rank = 3
    dim = res

    attr = {'Component_Rank':1, 'Component_Size':dim**rank, 'Dimensions':[dim]*rank, 'Rank':rank, 'TopGridDims':[dim]*rank, 'TopGridEnd':[dim]*rank, 'TopGridStart':[0]*rank}

    # renormalize so mean is 1
    dens = np.zeros([dim]*rank)
    xdim,ydim,zdim = dens.shape
    repsi = np.zeros([dim]*rank)
    impsi = np.zeros([dim]*rank)

    # set
    if (zero_momentum == True):
        side = np.linspace(0,256,256,endpoint=False)
        x,y,z = np.meshgrid(side,side,side,indexing='ij')
        
        # compute velocity
        vx0 = np.imag(np.gradient(psi,axis=0)*np.conjugate(psi))
        vy0 = np.imag(np.gradient(psi,axis=1)*np.conjugate(psi))
        vz0 = np.imag(np.gradient(psi,axis=2)*np.conjugate(psi))
        # center of mass velocity
        vbarx = np.sum(vx0)/np.sum(np.absolute(psi)**2)
        vbary = np.sum(vy0)/np.sum(np.absolute(psi)**2)
        vbarz = np.sum(vz0)/np.sum(np.absolute(psi)**2)

        print("initial COM velocity",vbarx,vbary,vbarz)

        # shift the phase to zero the COM velocity
        psi = psi*np.exp(-1j*(vbarx*x + vbary*y + vbarz*z))
    
        # check and diagnostic
        vx0 = np.imag(np.gradient(psi,axis=0)*np.conjugate(psi))
        vy0 = np.imag(np.gradient(psi,axis=1)*np.conjugate(psi))
        vz0 = np.imag(np.gradient(psi,axis=2)*np.conjugate(psi))
        vbarx = np.sum(vx0)/np.sum(np.absolute(psi)**2)
        vbary = np.sum(vy0)/np.sum(np.absolute(psi)**2)
        vbarz = np.sum(vz0)/np.sum(np.absolute(psi)**2)
        print("final COM velocity",vbarx,vbary,vbarz)
    
    repsi = np.real(psi)*np.sqrt(dens_to_enzo)
    impsi = np.imag(psi)*np.sqrt(dens_to_enzo)
    dens = np.absolute(psi)**2.0*dens_to_enzo
                
    if (absorption_boundary == True):
        side = np.linspace(0,dim,dim,endpoint=False)/dim-0.5
        x,y,z = np.meshgrid(side,side,side,indexing='ij')
        r = np.sqrt(x**2+y**2+z**2)

        dens = 1e2*np.arctan((r-0.48)*1e3)
        dens[r<0.48]=0.
    else:
        dens = np.absolute(psi)**2.0*dens_to_enzo

    # write out new density to new file
    f1 = h5.File('./GridDensity', 'w')
    new_dens = f1.create_dataset('GridDensity', data=dens)
    for a in attr:
        new_dens.attrs.create(a, attr[a])
    f1.close()

    # write out to new file
    f1 = h5.File('./GridRePsi', 'w')
    new_dens = f1.create_dataset('GridRePsi', data=repsi)
    for a in attr:
        new_dens.attrs.create(a, attr[a])
    f1.close()

    f1 = h5.File('./GridImPsi', 'w')
    new_dens = f1.create_dataset('GridImPsi', data=impsi)
    for a in attr:
        new_dens.attrs.create(a, attr[a])
    f1.close()

    enzo_len = 2 * boxedge * kpc_to_cm
    M_tot = np.sum(np.absolute(psi)**2) * (2 * boxedge / (res - 1))**3
    t_ff = t_freefall(boxedge, M_tot)

    f2 = open('halo_info.txt', 'w')
    f2.write('Boxside = '+str(boxedge*2)+' kpc\n')
    f2.write('Enzo LengthUnits = '+str(enzo_len)+'\n')
    f2.write('Total Mass = 10^'+str(np.log10(M_tot))+' solMass\n')
    f2.write('Freefall time = '+str(t_ff)+' Gyr ~ '+str(t_ff/8)+' Enzo TimeUnits\n')
    f2.close()

    return

    
def build_eigenmode_cubes(states, rsolve, R_nl, rcube, res, m_modes=None):

    x = np.linspace(-rcube, rcube, res)
    y = np.linspace(-rcube, rcube, res)
    z = np.linspace(-rcube, rcube, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    Rad = np.sqrt(X**2.0+Y**2.0+Z**2.0)
    phi = np.arctan2(Y,X)
    theta = np.arccos(Z/Rad)

    if m_modes is None:
        gridnum = len(states)
        print('No m_modes files provided; calculating only m=0 modes')
    else:
        gridnum = len(m_modes) - np.count_nonzero(m_modes) + np.sum(2 * states[m_modes][:, 0] + 1)

    psi = np.zeros((gridnum, res, res, res), dtype=np.complex64)
    indx = np.zeros((gridnum, 3))

    j = 0

    for i in range(len(states)):
        l = states[i, 0]
        n = states[i, 1]
        if m_modes is None:
            m = 0
            indx[i] = [l, n, m]
            RatRad = np.interp(Rad, rsolve, R_nl[i])
            spharm = scsp.sph_harm(m, l, phi, theta)
            psi[i] = (RatRad * spharm).astype(np.complex64)

        else:
            if m_modes[i]==False:
                m = 0
                indx[j] = [l, n, m]
                RatRad = np.interp(Rad, rsolve, R_nl[i])
                spharm = scsp.sph_harm(m, l, phi, theta)
                psi[j] = (RatRad * spharm).astype(np.complex64)
                j+=1
            else:
                for m in range(-l, l+1):
                    indx[j] = [l, n, m]
                    RatRad = np.interp(Rad, rsolve, R_nl[i])
                    spharm = scsp.sph_harm(m, l, phi, theta)
                    psi[j] = (RatRad * spharm).astype(np.complex64)

                    j += 1

    return psi, indx


def halo_decompose(psi, boxsize, boxedge, Mh0, l_max=None, n_max=None, den=None):
    """
    Decompose a halo into eigenmode amplitudes
    """
    if den==None:
        print("Function currently incomplete -- must include input density for now")
        return
    else:
        phi, rho = initiate_potential(den, Mh0, core=False)
        rsolve, E_nl, R_nl, shape_nl = compute_radial_solutions(phi, Mh0, verbose=True)

    x = np.linspace(-boxedge,boxedge,boxsize)
    y = np.linspace(-boxedge,boxedge,boxsize)
    z = np.linspace(-boxedge,boxedge,boxsize)
    X,Y,Z = np.meshgrid(x,y,z)
    Rad = np.sqrt(X**2.0+Y**2.0+Z**2.0)
    phi = np.arctan2(Y,X)
    theta = np.arccos(Z/Rad)

    binside = boxedge * 2.0 / boxsize

    if l_max==None:
        l_max = np.shape(E_nl)[0]
    if n_max==None:
        n_max = np.shape(E_nl)[1]

    A_nlm = np.zeros([l_max, n_max, 2 * l_max + 1])
    psi = np.zeros_like(Rad, dtype=complex)
    RatRad = np.zeros((n_max, np.shape(Rad)[0], np.shape(Rad)[1], np.shape(Rad)[2]))

    for l in range(l_max):
        print(l)

        for n in range(n_max):
            if E_nl[l, n] == 0.:
                continue
            else:
                RatRad[n] = np.interp(Rad, rsolve, R_nl[l, n, :len(rsolve)])

        for m in range(0, l+1):
            spharm_interp = scsp.sph_harm(m, l, phi, theta) 
            for n in range(n_max):
                if E_nl[l, n] == 0.:
                    continue
                else:
                    RatRad0 = RatRad[n]
                    if m==0:
                        psi0 = RatRad0 * spharm_interp
                        A_nlm[l, n, m] = np.sum(psi * psi0 * binside**3.)
                    else:
                        psi0 = RatRad0 * spharm_interp
                        A_nlm[l, n, m] = np.sum(psi * psi0 * binside**3.)
                        psi0 = RatRad0 * (-1)**m * np.conjugate(spharm_interp)
                        A_nlm[l, n, -m] = np.sum(psi * psi0 * binside**3.)
    return A_nlm


def evolve(rshell, E_nl, rsolve, R_nl, a_nlm, phase_nlm, t):
    # input: t in Gyr
    # m/h in code unit
    mhcoef = np.sqrt(m2h2)
    # 1Gyr in code unit
    tunit = 1e9*365*24*3600/(3.086e21/1e5)
    t = t*tunit
    phase_k = -E_nl*mhcoef*t
    phase = phase_nlm + phase_k[:,:,None] 
    return build_halo(rshell, E_nl, rsolve, R_nl, a_nlm, phase)


def find_vortices(psi,threshold=8):
    # velocity field
    vx = np.zeros(psi.shape)
    vy = np.zeros(psi.shape)
    vz = np.zeros(psi.shape)

    # vorticity field
    wx = np.zeros(psi.shape)
    wy = np.zeros(psi.shape)
    wz = np.zeros(psi.shape)
    
    psi = psi/np.absolute(psi)

    # compute velocity vector
    vx[1:-1,:,:] = np.imag((psi[2:,:,:]-psi[0:-2,:,:])*np.conjugate(psi[1:-1,:,:]))
    vy[:,1:-1,:] = np.imag((psi[:,2:,:]-psi[:,0:-2,:])*np.conjugate(psi[:,1:-1,:]))
    vz[:,:,1:-1] = np.imag((psi[:,:,2:]-psi[:,:,0:-2])*np.conjugate(psi[:,:,1:-1]))
    
    # compute vorticity vector
    wx[2:-2,2:-2,2:-2] = vz[2:-2,3:-1,2:-2] - vz[2:-2,1:-3,2:-2] - vy[2:-2,2:-2,3:-1] + vy[2:-2,2:-2,1:-3]
    wy[2:-2,2:-2,2:-2] = vx[2:-2,2:-2,3:-1] - vx[2:-2,2:-2,1:-3] - vz[3:-1,2:-2,2:-2] + vz[1:-3,2:-2,2:-2]
    wz[2:-2,2:-2,2:-2] = vy[3:-1,2:-2,2:-2] - vy[1:-3,2:-2,2:-2] - vx[2:-2,3:-1,2:-2] + vx[2:-2,1:-3,2:-2]
    
    curlv = abs(wx)**2 + abs(wy)**2 + abs(wz)**2
                        
    xs,ys,zs = np.where(curlv>threshold)
    return xs,ys,zs


def plot_vortices(psi):
    # find points on the vortex lines
    xs,ys,zs = find_vortices(psi)
    # glue points on the same vortex line together
    data = np.array(list(zip(xs,ys,zs)))
    clustering = DBSCAN(eps=2.5, min_samples=2).fit(data)

    cmap=plt.get_cmap('tab10')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')


    ax.scatter(xs, ys, zs, s=0.2,c=cmap(np.mod(clustering.labels_,10)),rasterized=True)

    ax.view_init(30, 30)

    ax.set_xlabel(r'$x$', fontsize=24)
    ax.set_ylabel(r'$y$', fontsize=24)
    ax.set_zlabel(r'$z$', fontsize=24)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
