import numpy as np
import scipy as sp
import scipy.interpolate as scinterp
import scipy.integrate as scinteg
import warnings


# Gravitational constant in kpc / M_sun * (km / s)^2
G = 4.30091727e-6


def phi_calc(r0, rho):
    """
    Compute the gravitational potential at a radius given the density, using Poisson's equation
    
    Parameters
    ----------
    r0 : numeric
        Radius (in kpc) at which to calculate the gravitational potential
    rho : lambda function or spline object
        Lambda function or cubic spline object that returns the value of the density (in solar mass per kpc^3) as a function of radius (in kpc)
    
    Returns
    -------
    phi : numpy array
        Value of the gravitational potential corresponding to the input radius in (km / s)^2

    """
    integrand1 = lambda r: rho(r) * r**2
    integrand2 = lambda r: rho(r) * r

    warnings.filterwarnings("ignore")

    integral1, err = scinteg.quad(integrand1, 0., r0)
    integral2, err = scinteg.quad(integrand2, r0, np.inf)

    return -4. * np.pi * G * (1. / r0 * integral1 + integral2)


def initialize_potential(rho_in, rmin=0.001, rmax=1000.):
    """
    Initialize functions for calculating the potential and density as a function of radius
    
    Parameters
    ----------
    rho_in : function
        A Python function that returns the density in Solar Masses per kpc^3 at a given radius (in kpc)
    rmin : numeric (optional, default: 0.001)
        Minimum radius, in kpc, for computation of density (for r<rmin, density is assumed to be equal to density at rmin)
    rmax : numeric (optional, default: 1000.)
    	Maximum radius, in kpc, for computation of density (for r>rmax, density is assumed to be zero)
    
    Returns
    -------
    phi_out : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the potential in (km / s)^2 as a function of radius (in kpc)
    rho_out : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the density in Solar Masses per kpc^3 as a function of radius
    M_total : numeric
    	Total mass of the halo out to rmax in solar masses

    """
    r_spline = np.logspace(np.log10(rmin), np.log10(rmax), 256)
    rho_halo = rho_in(r_spline)

    rho_out = scinterp.interp1d(r_spline, rho_halo, kind='cubic', bounds_error=False, fill_value=(rho_halo[0], 0.))

    phi_spline = np.array([phi_calc(r, rho_in) for r in r_spline])
    phi_out = scinterp.interp1d(r_spline, phi_spline, kind='cubic', bounds_error=False, fill_value='extrapolate')

    dM = lambda r: 4. * np.pi * rho_out(r) * r**2
    M_total, err = scinteg.quad(dM, 0., rmax)

    return phi_out, rho_out, M_total


def DF_invert(phi, rho, DF_type='Isotropic', rmin=0.001, rmax=1000.):
    """
    Inversion formula to calculate the distribution function f(E) of a potential given an array of total energy

    Parameters
    ----------
    phi : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the potential in (km / s)^2 as a function of radius (in kpc)
    rho : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the density in Solar Masses per kpc^3 as a function of radius
    DF_type : string (optional, default: 'Isotropic')																	TO IMPLEMENT!!!
    	string specifying whether the returned distribution function should be isotropic or radial
    rmin : numeric (optional, default: 0.001)
        Minimum radius, in kpc, for computation of density (for r<rmin, density is assumed to be equal to density at rmin)
    rmax : numeric (optional, default: 1000.)
    	Maximum radius, in kpc, for computation of density (for r>rmax, density is assumed to be zero)
    
    Returns
    -------
    f_E : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the distribution function as a function of the total energy in (km / s)^2

    """
    r_spline = np.logspace(np.log10(rmin), np.log10(rmax), 256)
    phi_spline = phi(r_spline)
    rho_spline = rho(r_spline)
    idx = np.argsort(phi_spline)

    e_range = phi_spline[idx]

    r_phi = scinterp.CubicSpline(phi_spline[idx], r_spline[idx])
    rho_phi = scinterp.CubicSpline(phi_spline[idx], rho_spline[idx])
    drho_dphi = rho_phi.derivative()
    
    if DF_type=='Isotropic':

        def integrand(phi_integ, e_prime):
            return 1. / np.sqrt(phi_integ - e_prime) * drho_dphi(phi_integ)

        prefactor = 1. / (2. * np.sqrt(2.) * np.pi**2)
        
    elif DF_type=='Radial':
        def integrand(phi_integ, e_prime):
            return 1. / np.sqrt(phi_integ - e_prime) * rho_phi(phi_integ) * r_phi(phi_integ)**2

        prefactor = -1. / (np.sqrt(2.) * np.pi**2)
        
    else:
        print("DF_type must be 'Isotropic' or 'Radial'")
        return None

    integ_spline = np.array([scinteg.quad(integrand, e_prime, e_range[-1], args=(e_prime,))[0] for e_prime in e_range[:-1]])

    integral = scinterp.CubicSpline(e_range[:-1], prefactor * integ_spline)
    f_E = integral.derivative()

    return f_E


def g_E(phi, rmin=0.001, rmax=1000.):
    """
    Determination of the classical density of states (typically labeled g(E)) given an input potential 

    Parameters
    ----------
    phi : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the potential in (km / s)^2 as a function of radius (in kpc)
    rmin : numeric (optional, default: 0.001)
        Minimum radius, in kpc, for computation of density (for r<rmin, density is assumed to be equal to density at rmin)
    rmax : numeric (optional, default: 1000.)
    	Maximum radius, in kpc, for computation of density (for r>rmax, density is assumed to be zero)

    Returns
    -------
    g_E : scipy ppoly object (cubic spline)
        Cubic spline object that returns the value of the density of states as a function of the total energy in (km / s)^2

    """
    r_spline = np.logspace(np.log10(rmin), np.log10(rmax), 256)
    phi_spline = phi(r_spline)
    phi_ppoly = scinterp.CubicSpline(r_spline, phi_spline)
    Emin = phi(rmin)
    Emax = phi(rmax)
    Espline = np.linspace(Emin, Emax, 1024)

    def integrand(r, Einteg):
        return r**2 * np.sqrt(2. * (Einteg - phi(r)))

    g_Espline = np.array([(4. * np.pi)**2 * scinteg.quad(integrand, 0., phi_ppoly.solve(E, extrapolate=False)[0], args=(E,))[0] for E in Espline])

    gE = scinterp.CubicSpline(Espline, g_Espline)

    return gE
