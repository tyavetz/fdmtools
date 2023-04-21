import numpy as np
import healpy as hp


G = 4.30091727e-6						# Gravitational constant in kpc / M_sun * (km / s)^2



def halo_density(r_array, E_nl, R_nl, a_nlm, phase_nlm, res=None, verbose=False):
    """
	Build a FDM halo on a spherical healpix grid given input from the construction module

    Parameters
    ----------
    TO DO
    
    Returns
    -------
    TO DO

    """

    if res==None:
        res = r_array[1] - r_array[0]
    pix_res = np.rint(4. * np.pi * r_array[-1]**2 / res**2).astype(int)
    nside = hp.pixelfunc.get_min_valid_nside(pix_res)
    npix = hp.nside2npix(nside)
    rho_lms_size = hp.Alm.getsize(3 * nside - 1)

    steps = len(r_array)
    grid = np.zeros((steps, npix))
    rho_lms = np.zeros((steps, rho_lms_size), dtype=complex)

    nmax, lmax = np.shape(a_nlm)

    if verbose==True:
        print('Progress:')

    for i in range(steps):
        if i%20==0:
            if verbose==True:
                print('Shell '+str(i)+' out of '+str(steps))
        a_nlm_real = np.zeros((lmax+2)*(lmax+1)//2, dtype=complex)
        a_nlm_imag = np.zeros((lmax+2)*(lmax+1)//2, dtype=complex)

        for l in range(lmax):
            for m in range(0, l+1):
                AR = 0.
                idx = (2*lmax+3-m)*m//2 + (l-m)
                for n in range(nmax):
                    if E_nl[n, l] != 0.:
                        AR = a_nlm[n, l] * R_nl[n, l](r_array[i])
                        alm = np.cos(phase_nlm[n, l, m]) + 1.j * np.sin(phase_nlm[n, l, m])
                        almm = (np.cos(phase_nlm[n, l, -m]) - 1.j * np.sin(phase_nlm[n, l, -m])) * (-1.)**m
                        if m==0:
                            a_nlm_real[idx] += AR * alm
                            a_nlm_imag[idx] += AR * alm / (1.j)
                        else:
                            a_nlm_real[idx] += AR / 2. * (alm + almm)
                            a_nlm_imag[idx] += AR / 2.j * (alm - almm)
            
        new_map_real = hp.alm2map(a_nlm_real, nside=nside)
        new_map_imag = hp.alm2map(a_nlm_imag, nside=nside) * 1.j

        new_map = new_map_real + new_map_imag
            
        grid[i] = np.abs(new_map)**2
        rho_lms[i] = hp.map2alm(grid[i])


    return rho_lms, grid



def halo_potential(r_array, rho_lms, verbose=False):
    """
    Calculate the gravitational potential spherical harmonic transform using variation of parameters

    Parameters
    ----------
    TO DO
    
    Returns
    -------
    TO DO

    """

    lmax_phi = hp.Alm.getlmax(np.shape(rho_lms[0])[0])
    print(lmax_phi)

    phi_lms = np.zeros_like(rho_lms)
    ls = hp.Alm.getlm(lmax_phi)[0][None,:]
    log_integrand1 = (1. - ls) * np.log(r_array[:,None]) + np.log(rho_lms)
    log_integrand2 = (2. + ls) * np.log(r_array[:,None]) + np.log(rho_lms)

    steps = len(r_array)

    if verbose==True:
        print('Progress:')

    for i in range(steps):
        if i%20==0:
            if verbose==True:
                print('Shell '+str(i)+' out of '+str(steps))
        integral1 = np.trapz(np.exp(ls[0] * np.log(r_array[i]) + log_integrand1[i:]), x=r_array[i:], axis=0)
        integral2 = np.trapz(np.exp((-ls[0] - 1) * np.log(r_array[i]) + log_integrand2[:i+1]), x=r_array[:i+1], axis=0)
        phi_lms[i] = integral1 + integral2
    phi_lms *= -4. * np.pi * G / (2. * ls[0] + 1)

    return phi_lms


