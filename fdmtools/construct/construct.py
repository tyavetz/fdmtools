import numpy as np
import scipy as sp
import scipy.interpolate as scinterp
import scipy.integrate as scinteg
import warnings


G = 4.30091727e-6						# Gravitational constant in kpc / M_sun * (km / s)^2
hbar = 6.582119569509067e-16    		# Reduced Planck constant in eV * s
c_constant = 2.99792458e5       		# Speed of light in km/s

km_to_kpc = 3.240779289444365e-17       # Conversion from km to kpc
km_s_to_kpc_gyr = 1.022712165045695     # Conversion from km/s to kpc/Gyr



def h_over_m(m_a):
    """
	Returns value of h over the mass of the axion (in eV / c^2).

    Parameters
    ----------
    TO DO
    
    Returns
    -------
    TO DO

    """

    return hbar / (m_a / c_constant**2) * km_to_kpc


def valsvecs(axion_mass, rmax, nsteps, l, phi, Emax, overshoot=1.2):
    """
	Returns the eigenvalues and eigenvectors of the Schrodinger equation for a given value of l, a gravitational potential, and a maximum total energy,
    Uses the finite difference method. 

    Parameters
    ----------
    TO DO
    
    Returns
    -------
    TO DO

    """

    h_a = h_over_m(axion_mass)

    r_array_temp, step = np.linspace(0., rmax * overshoot, nsteps+1, retstep=True)
    r_array = r_array_temp[1:]

    # TO DO: validate that r_array gives good enough resolution (or does steps parameter need to be increased)

    if l==0:
    	print('Sampling grid for eigenvalue problem includes '+str(nsteps)+' bins, out to r_max * '+str(overshoot)+' = '+str(round(rmax * 1.2, 2))+' kpc')
    	print('(returning only eigenmodes contained within r_max = '+str(round(rmax, 2))+' kpc)')

    Dr2_0 = -h_a**2 / (2 * step**2) * np.ones(nsteps) * (-2.)
    Dr2_1 = -h_a**2 / (2 * step**2) * np.ones(nsteps - 1)
    Vvec = h_a**2 * l * (l + 1) / (2. * r_array**2) + phi(r_array)
        
    E_n, u_vecs_T = sp.linalg.eigh_tridiagonal(Dr2_0 + Vvec, Dr2_1, select='v', select_range=(-np.inf, Emax))
    u_vecs = u_vecs_T.T
    
    R_n = np.empty((1, len(E_n)), dtype=object)
    
    for i in range(len(E_n)):

        R0 = u_vecs[i] / r_array
        A = 1. / np.sqrt(np.trapz(R0**2. * r_array**2., r_array))
        R1 = R0 * A
        R_n[0, i] = scinterp.CubicSpline(r_array, R1)

        Mass_in = np.trapz(R0[r_array<=rmax]**2. * r_array[r_array<=rmax]**2., r_array[r_array<=rmax])
        Mass_out = np.trapz(R0[r_array>rmax]**2. * r_array[r_array>rmax]**2., r_array[r_array>rmax])

        imax = i

        if Mass_out > Mass_in * 0.01:
            break

    E_n = E_n[:imax]
    R_n = R_n[:, :imax]

    if np.all(np.diff(E_n, n=2)[:-1] < 1e-8)==False:
        print('Warning! There may be missing and/or duplicated l='+str(l)+' eigenvalues; insufficient resolution')

    return E_n, R_n



def compute_radial_solutions(axion_mass, phi, rho, rmax, nsteps, overshoot=1.2, verbose=True):
    """
    Compute all eigenvalues and eigenvectors (for all l) for the Schrodinger equation

    Parameters
    ----------
    TO DO
    
    Returns
    -------
    TO DO

    """

    dM = lambda r: 4. * np.pi * rho(r) * r**2
    M_fit, err = scinteg.quad(dM, 0., rmax)
    Ek = 0.5 * G * M_fit / rmax
    Emax = phi(rmax) + Ek

    E_nl_grid = []
    l = 0
    shape_n = 1
    total_modes_nl = 0
    total_modes_nlm = 0
    while shape_n>0:
        E_n, R_n = valsvecs(axion_mass, rmax, nsteps, l, phi, Emax, overshoot=overshoot)
        shape_n = len(E_n)
        if shape_n==0:
            break
        if l==0:
            R_nl_grid = np.copy(R_n.T)
            n_max = shape_n

        elif shape_n == n_max:
            R_nl_grid = np.hstack((R_nl_grid, R_n.T))

        else:
            R_n_long = np.concatenate((R_n, np.empty((1, n_max - shape_n), dtype=object)), axis=1)
            R_nl_grid = np.hstack((R_nl_grid, R_n_long.T))

        E_nl_array = np.zeros(n_max)
        E_nl_array[:shape_n] = E_n

        E_nl_grid.append(E_nl_array)

        if verbose==True:
            print('l = '+str(l)+'; n_max = '+str(shape_n))
        total_modes_nl += shape_n
        total_modes_nlm += shape_n * (2 * l + 1)
        l+=1
    l_max = l

    print('n_max = '+str(n_max)+'; l_max = '+str(l_max)+'; distinct nl eigenmodes = '+str(total_modes_nl)+'; total eigenmodes = '+str(total_modes_nlm))

    return np.array(E_nl_grid).T, R_nl_grid



def amp_from_fE(axion_mass, E_nl, fE, Type='Isotropic'):
    """
    Compute the eigenmode amplitudes a_nlm directly from an input distribution function

    Parameters
    ----------
    TO DO
    
    Returns
    -------
    TO DO

    """

    h_a = h_over_m(axion_mass)

    if Type=='Radial':
        factor = np.sqrt((2. * np.pi)**3. * h_a)
        E_nl[:, 1:] = 0.
    elif Type=='Isotropic':
        factor = np.sqrt((2. * np.pi * h_a)**3.)
    else:
        raise RuntimeError("Type must be either 'Isotropic' or 'Radial'")

    a_nlm = np.zeros_like(E_nl)

    a_nlm[E_nl!=0.] = np.nan_to_num(np.sqrt(fE(E_nl[E_nl!=0.]))) * factor

    return a_nlm


def random_phase(nmax, lmax):
    """
    Output a random phase for each eigenmode (n, l, and m) given lmax and nmax

    Parameters
    ----------
    TO DO
    
    Returns
    -------
    TO DO
    """

    phase_nlm = np.random.rand(nmax, lmax, 2 * lmax + 1) * 2. * np.pi
    
    return phase_nlm


def evolve_phase(axion_mass, phase_nlm, E_nl, t_array):
	"""
    Evolve the phase given E_nl over an array of times t_array (in Gyr)

    Parameters
    ----------
    TO DO
    
    Returns
    -------
    TO DO
    """

	h_a = h_over_m(axion_mass)
	phase_k = -E_nl[:, :, None, None] / h_a * km_s_to_kpc_gyr * t_array[None, :]
	phase = phase_nlm[:, :, :, None] + phase_k
	return phase



