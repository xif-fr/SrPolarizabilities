# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 15:52:37 2021.

@author: Romaric
"""

import numpy as np
import pandas as pd
# import openpyxl

from sympy.physics.wigner import wigner_3j, wigner_6j
from PhysicalConstants import ReducedPlanckConstant as hbar
from PhysicalConstants import SpeedOfLight as c
from PhysicalConstants import VacuumPermittivity as eps0
from math import sqrt, pi

transition0 = transition1 = pd.read_csv('sr_new.csv', sep='\t')

# =============================================================================
# By convention B-field will impose the quantification axis to be along z-axis
# The formulas are taken from Le Kien et al.: Dynamical polarizability of atoms
# in arbitrary light fields: general theory and application to cesium
# =============================================================================

alpha_atomic_unit = 1.648_777_274e-41  # C^2.m^2.J^-1
d_atomic_unit = 8.478_353_625_5e-30

# Contributions to the scalar polarizabilities of the different levels in a.u.
# taken from tables from Safronova from different articles referenced in the 
# labbook. 
core_polarizabilities = {'5s2-1S0': 5.3,
                         '5s5p-3P0': 5.6,
                         '5s5p-3P1': 5.6,
                         '5s5p-3P2': 5.6}


def W_3j(j1, j2, j3, m1, m2, m3):
    """
    Wigner-3j symbol from Sympy converted into float.

    Parameters
    ----------
    j1 : int
    j2 : int
    j3 : int
    m1 : int
    m2 : int
    m3 : int

    Returns
    -------
    float
        Wigner-3j symbol.

    """
    return float(wigner_3j(j1, j2, j3, m1, m2, m3))


def W_6j(j1, j2, j3, j4, j5, j6):
    """
    Wigner-6j symbol from Sympy converted into float.

    Parameters
    ----------
    j1 : int
    j2 : int
    j3 : int
    j4 : int
    j5 : int
    j6 : int

    Returns
    -------
    float
        Wigner-6j symbol.

    """
    return float(wigner_6j(j1, j2, j3, j4, j5, j6))


class energy_level:
    """Class for atomic energy levels labeled by n, J, M_J."""

    def __init__(self, configuration, term, J, gamma=0, M=None):
        self.configuration = configuration
        self.term = term
        self.J = J
        self.M = M
        self.gamma = gamma
        self.name = configuration + '-' + term + str(J)


# Equation 14
def d_red(omega, Ak, Jk):
    """
    Compute reduced dipole matrix element.

    Parameters
    ----------
    omega : float
        Bohr angular frequency of the transition.
    Ak : float
        Einstein A coefficient.
    Jk : int
        Total angular momentum J = L + S.

    Returns
    -------
    float
        Reduced dipole in C.m (SI).

    """
    return sqrt(3 * pi * eps0 * hbar * c**3 * Ak * (2 * Jk + 1) / omega**3)


# Equation 12
def polar_tensor_K_q(u, K, q):
    """
    Compute q component of tensor part of rank K from the polarization u.

    Parameters
    ----------
    u : list
        polarization in spherical coordinates.
    K : int
        rank of the tensor K = 0, 1, 2.
    q : int
        q = -K, ..., K

    Returns
    -------
    tensor : float

    """
    tensor = 0
    for mi in [-1, 0, 1]:
        for mk in [-1, 0, 1]:
            tensor += ((-1)**(q + mk) * u[mi + 1] * np.conj(u[-(mk + 1) + 2]) *
                       sqrt(2 * K + 1) *
                       W_3j(1, K, 1,
                            mi, -q, mk)
                       )
    return tensor


# Equation 11: Reduced dynamical polarizabilties
def alpha_nJ_K(initial_level, transition, K, omega_laser, save=False):
    """
    Compontent of polarizability.

    Parameters
    ----------
    initial_level : dict
        Fine-structure level.
    transitions : dataframe
        Transitions over which the summation will be performed
    K : int
        0 for scalar
        1 for vector
        2 for tensor.

    Returns
    -------
    alpha : float
        reduced dynamical polarizability for nJ and K.

    """
    tr = transition

    conf_i = initial_level.configuration
    term_i = initial_level.term
    J = initial_level.J
    # gamma_i = initial_level.gamma
    # state_i = initial_level.name

    # Adding by hand the transition from 3P1 to 1S0
    # tr = tr.append(dict(zip(tr.columns, 
    #                         ['5s5p', '3P', '1', 14504,
    #                           '5s2', '1S', 0, 0, 0.151, 'NIST']
    #                         )), ignore_index=True)

    # filtering the data
    tr = tr[((tr['conf_i'] == conf_i) & (tr['term_i'] == term_i) & (tr['J_i'] == J))
            | ((tr['conf_k'] == conf_i) & (tr['term_k'] == term_i) & (tr['J_k'] == J))]



    # print(tr)
    alpha = 0
    table_alpha = pd.DataFrame(columns=['Initial', 'Final', 'K', 'alpha_k'])
    # print(table_alpha)
    # print(f'Table of polarizability of rank {K} for state {state}')
    for index, row in tr.iterrows():

        tr_k = row
        Ji = tr_k['J_i']
        Jk = tr_k['J_k']
        omega_i = 2 * pi * c * tr_k['E_i (cm-1)'] * 1e2
        omega_k = 2 * pi * c * tr_k['E_k (cm-1)'] * 1e2
        D_red = tr_k['D (a.u.)'] * d_atomic_unit
        state_k = tr_k['conf_k'] + '-' + tr_k['term_k'] + str(tr_k['J_k'])
        state_i = tr_k['conf_i'] + '-' + tr_k['term_i'] + str(tr_k['J_i'])
        # Test without imaginary part
        gamma = 0
        # print(f'gamma = {gamma:.0f}')
        if state_i != initial_level.name:
            omega_i, omega_k = (omega_k, omega_i)
            Ji, Jk = (Jk, Ji)
        
        alpha_k = ((-1)**(K + Ji + 1) * sqrt(2 * K + 1) *
                   (-1)**Jk * W_6j(1, K, 1,
                                   Ji, Jk, Ji) *
                   np.abs(D_red)**2 *
                   1 / hbar *
                   np.real(1 / ((omega_k - omega_i) - omega_laser - 1j * gamma / 2) +
                           (-1)**K / ((omega_k - omega_i) + omega_laser + 1j * gamma / 2))
                   )

        alpha += alpha_k
        # print(state_i, state_k)
        # print('Final state: ' + state_k)
        # print(alpha_k)
        # print([state_i, state_k, K, alpha_k])
        if save:
            df_k = pd.DataFrame(columns=['Initial', 'Final', 'K', 'alpha_k'],
                                data=[[state_i, state_k, K, alpha_k]])
            # print('df_k', df_k)
            table_alpha = table_alpha.append(df_k)
        # print(table_alpha)

        # table_alpha.to_csv(f'tables/alphas_{K}_' + state_i + '.csv',
        #                     sep=',')
        # table_alpha.to_csv(f'tables/alphas_{K}_' + state_i + '.xlsx')
    
            # path = 'tables_1064_nm/alphas_' + initial_level.name + '_test' + '.xlsx'
            # book = openpyxl.load_workbook(path)
    
            # with pd.ExcelWriter(path, mode='a') as writer:
            #     writer.book = book
            #     writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
            #     table_alpha.to_excel(writer, sheet_name=f'rank {K}')
    return alpha


# Equation 16: scalar, vector and tensor polarizabilities
def polarizabilities(initial_level, transition, omega_laser):
    """
    Decomposition of polarizability tensor into scalar, vector, tensor parts.

    Parameters
    ----------
    initial_level : dict
        Fine-structure level.
    transitions : dict
        Transitions over which the summation will be performed


    Returns
    -------
    alpha : float
        reduced dynamical polarizability for nJ and K.

    """
    J = initial_level.J
    alpha_s = (1 / sqrt(3 * (2 * J + 1))
               * alpha_nJ_K(initial_level, transition, 0, omega_laser))
    alpha_v = (- sqrt((2 * J) / ((J + 1) * (2 * J + 1)))
               * alpha_nJ_K(initial_level, transition, 1, omega_laser))
    alpha_t = (- sqrt((2 * J * (2 * J - 1))
                      / (3 * (J + 1) * (2 * J + 1) * (2 * J + 3)))
               * alpha_nJ_K(initial_level, transition, 2, omega_laser))
    
    if initial_level.name in core_polarizabilities.keys():
        alpha_s += core_polarizabilities[initial_level.name] * alpha_atomic_unit
        
    return alpha_s, alpha_v, alpha_t


# Equation 19 and 20
def AC_Stark_shift_strong_B(E0, epsilon, initial_level,
                            transition, omega_laser):
    """
    Compute AC Stark shift in the limit of strong bias field along z.

    Parameters
    ----------
    E0 : Tfloat
        Amplitude of the electric field in V/m.
    epsilon : list(complex)
        Three cartesian components of the polarization unit vector.
    initial_level : instance of energy_level
        State on which the shift is applied.
    transition : dict
        Atomic data from AtomicData.py.
    omega_laser : float
        Angular frequency of the laser.

    Returns
    -------
    L_shifts : list(float)
        List of the shifts on the different Zeeman substates.

    """
    J = initial_level.J
    ux, uy, uz = epsilon
    C = 2 * np.imag(np.conj(ux) * uy)
    D = 1 - 3 * np.abs(uz)**2
    alpha_s, alpha_v, alpha_t = polarizabilities(initial_level,
                                                 transition, omega_laser)    
    L_shifts = []
    print('\nAC Stark shift in strong B-field for '
          + initial_level.name + f'\nepsilon = {epsilon}' + '\n')
    print(f'[alpha_s, alpha_v, alpha_t] = [{alpha_s:.2e}, {alpha_v:.2e}, '
          f'{alpha_t:.2e}] SI')
    print(f'[alpha_s, alpha_v, alpha_t] = [{alpha_s/alpha_atomic_unit:.0f}, '
          f'{alpha_v/alpha_atomic_unit:.0f}, '
          f'{alpha_t/alpha_atomic_unit:.0f}] a.u.' + '\n')
    if J != 0:

        for M in range(-J, J + 1):

            Shift = (-1/4 * E0**2 *
                     (alpha_s +
                      alpha_v * C * M / (2 * J) -
                      alpha_t * D *
                      (3 * M**2 - J * (J + 1))/(2 * J * (2 * J - 1))
                      )
                     )
            print(f'Shift for M = {M}: {Shift:.2e} J')
            print(f'Shift for M = {M}: {Shift / (hbar * 2 * pi):.2e} Hz\n')
            L_shifts.append(Shift)
    else:
        Shift = -1/4 * E0**2 * alpha_s
        print(f'Shift : {Shift:.2e} J')
        print(f'Shift : {Shift / (hbar * 2 * pi):.2e} Hz\n')
        L_shifts.append(Shift)
    return L_shifts

def plot_alpha(initial_level, transition, lambdas):
    import matplotlib.pyplot as plt
    import pandas as pd
    alpha_s = []
    alpha_v = []
    alpha_t = []
    for lbd in lambdas:
        omega_laser = 2 * pi * c / lbd
        a_s, a_v, a_t = polarizabilities(initial_level, transition, omega_laser)
        alpha_s.append(a_s/alpha_atomic_unit)
        alpha_v.append(a_v/alpha_atomic_unit)
        alpha_t.append(a_t/alpha_atomic_unit)
    
    alphas = np.array(alpha_s)
    
    fig, ax = plt.subplots(1, 1,
                            sharex=True,
                            sharey=True,
                            )
    ax.set_xlabel(r'Wavelength $\lambda$ (nm)')
    ax.set_ylabel('Polarizabilities in a.u.')
    
    ax.plot(lambdas * 1e9, alpha_s, label=r'$\alpha_s$')
    if initial_level.J != 0:
        ax.plot(lambdas * 1e9, alpha_v, label=r'$\alpha_v$')
        ax.plot(lambdas * 1e9, alpha_t, label=r'$\alpha_t$')
    

    plt.title(f'Polarizabilities for {initial_level.configuration + initial_level.term}')
    plt.legend()
    plt.show()
    
    df = pd.DataFrame({'Wavelength (m)':lambdas,
                       'Scalar polarizability (a.u.)': alpha_s,
                       'Vector polarizability (a.u.)': alpha_v,
                       'Tensor polarizability (a.u.)': alpha_t
                       })
    
    return df

def plot_alpha_DC(initial_level, transition, lambdas=np.logspace(-6, 0, 60)):
    import matplotlib.pyplot as plt
    import pandas as pd
    alpha_s = []
    alpha_v = []
    alpha_t = []
    for lbd in lambdas:
        omega_laser = 2 * pi * c / lbd
        a_s, a_v, a_t = polarizabilities(initial_level, transition, omega_laser)
        alpha_s.append(a_s/alpha_atomic_unit)
        alpha_v.append(a_v/alpha_atomic_unit)
        alpha_t.append(a_t/alpha_atomic_unit)
    
    alphas = np.array(alpha_s)
    
    fig, ax = plt.subplots(1, 1,
                            sharex=True,
                            sharey=True,
                            )
    ax.set_xlabel(r'Wavelength $\lambda$ (nm)')
    ax.set_ylabel(r'Polarizabilities in a.u.')
    
    ax.semilogx(lambdas * 1e9, alpha_s, label=r'$\alpha_s$')
    if initial_level.J != 0:
        ax.semilogx(lambdas * 1e9, alpha_v, label=r'$\alpha_v$')
        ax.semilogx(lambdas * 1e9, alpha_t, label=r'$\alpha_t$')
    

    plt.title(rf'Polarizabilities for {initial_level.configuration} {initial_level.term}' +
              rf'\n $a_s(\omega=0) \simeq {alpha_s[-1]:.0f}a.u.$')
    plt.legend()
    plt.show()
    
    df = pd.DataFrame({'Wavelength (m)':lambdas,
                       'Scalar polarizability (a.u.)': alpha_s,
                       'Vector polarizability (a.u.)': alpha_v,
                       'Tensor polarizability (a.u.)': alpha_t
                       })
    
    return df

if __name__ == '__main__':

    ground = energy_level(configuration='5s2', term='1S', J=0)
    excited = energy_level(configuration='5s5p', term='3P', J=1)
    metastable = energy_level(configuration='5s5p', term='3P', J=2)
    lambda_laser_nm = float(input("Wavelength (nm): "))
    lambda_laser = lambda_laser_nm * 1e-9
    omega_laser = 2 * pi * c / (lambda_laser)

    print(f'{lambda_laser = }m')
    alphag_scalar, _, _ = polarizabilities(ground, transition0, omega_laser)
    print(f'g scalar polarizability {alphag_scalar/alpha_atomic_unit:.2f} a.u.')
    alphae_scalar, alphae_vector, alphae_tensor = polarizabilities(excited, transition1, omega_laser)
    print(f'e scalar polarizability {alphae_scalar/alpha_atomic_unit:.2f} a.u.')
    print(f'e vector polarizability {alphae_vector/alpha_atomic_unit:.2f} a.u.')
    print(f'e tensor polarizability {alphae_tensor/alpha_atomic_unit:.2f} a.u.')

    print()
    print(f'δαs = {(alphae_scalar-alphag_scalar)/alpha_atomic_unit:.2f} a.u.')
    print(f'δαs-αt/2 = {(alphae_scalar-alphag_scalar-alphae_tensor/2)/alpha_atomic_unit:.2f} a.u.')
    print(f'δαs-αt/2-αv/2 = {(alphae_scalar-alphag_scalar-alphae_tensor/2-alphae_vector/2)/alpha_atomic_unit:.2f} a.u.')
    print(f'δαs+αt = {(alphae_scalar-alphag_scalar+alphae_tensor)/alpha_atomic_unit:.2f} a.u.')
    print(f'δαs-2αt = {(alphae_scalar-alphag_scalar-2*alphae_tensor)/alpha_atomic_unit:.2f} a.u.')

    print()
    print(f'rel. αs = {alphae_scalar/alphag_scalar:.3f}')
    print(f'rel. αv = {alphae_vector/alphag_scalar:.3f}')
    print(f'rel. αt = {alphae_tensor/alphag_scalar:.3f}')
    print(f'rel. αs-αt/2 = {(alphae_scalar-alphae_tensor/2)/alphag_scalar:.3f}')
    print(f'rel. αs+αt = {(alphae_scalar+alphae_tensor)/alphag_scalar:.3f}')
    print(f'rel. αs-2αt = {(alphae_scalar-2*alphae_tensor)/alphag_scalar:.3f}')
    print(f'rel. αs-αt/2+|αv|/2 = {(alphae_scalar-alphae_tensor/2+np.abs(alphae_vector)/2)/alphag_scalar:.3f}')
    print(f'rel. αs-αt/2-|αv|/2 = {(alphae_scalar-alphae_tensor/2-np.abs(alphae_vector)/2)/alphag_scalar:.3f}')
    print(f'rel. αs-αt/8+|αv|/2√2 = {(alphae_scalar-alphae_tensor/8+np.abs(alphae_vector)/2/np.sqrt(2))/alphag_scalar:.3f}')

    #--------------

    # print(f'scalar polarizability {alpha_scalar:.2e} in SI units')
    # print(f'scalar polarizability {alpha_scalar/alpha_atomic_unit:.1f} in a.u.')
    # print('scalar polarizability from PRX 8,041055 (2018): 878.0 a.u. @520nm\n'
    #       + 'discrepancy coming from 1P1 contribution mainly')

    # print()

    # I_laser = 1.0 * 1e6  #  W/m² (= 1W/mm²)
    # print(f'\nI_laser = {I_laser:.1e} W/m²')
    # Delta_E_unitaire = -1/4 * alpha_scalar * (I_laser / (eps0 * c / 2))
    # print(f'Delta_E_unitaire = {Delta_E_unitaire:.2e} J')
    # print(f'Delta_E_unitaire = {Delta_E_unitaire / (hbar * 2 * pi):.2e} Hz')

    # E0 = sqrt(I_laser / (eps0 * c / 2))

    # epsilon_z = [0, 0, 1]
    # AC_Stark_shift_strong_B(E0, epsilon_z, ground, transition, omega_laser)
    # AC_Stark_shift_strong_B(E0, epsilon_z, excited, transition, omega_laser)

    # epsilon_x = [1, 0, 0]
    # AC_Stark_shift_strong_B(E0, epsilon_x, ground, transition, omega_laser)
    # AC_Stark_shift_strong_B(E0, epsilon_x, excited, transition, omega_laser)

    # AC_Stark_shift_strong_B(E0, epsilon_x, metastable,
    #                         transition, omega_laser)
    # epsilon_tilt = [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
    # AC_Stark_shift_strong_B(E0, epsilon_tilt, ground,
    #                         transition, omega_laser)
    # AC_Stark_shift_strong_B(E0, epsilon_tilt, excited,
    #                         transition, omega_laser)
    
    # epsilon_sigma = [1/sqrt(2), 1j * 1/2, -1j * 1/2]
    # AC_Stark_shift_strong_B(E0, epsilon_sigma, ground,
    #                         transition, omega_laser)
    # AC_Stark_shift_strong_B(E0, epsilon_sigma, excited,
    #                         transition, omega_laser)

    # ---------
    
    # e1P1 = energy_level(configuration='5s5p', term='1P', J=1)
    # omega_at = w0 = 2 * pi * c / 461e-9
    # omega_laser = w = 2 * pi * c / 532e-9
    # Gamma_sc_532 = 3*pi*c**2 / (2*hbar*w0**3) * (w/w0)**3 * (31e6 * 2 * pi)**2 *(1/(w0-w)+1/(w0+w))**2
    
    # omega_laser_1064 = w = 2 * pi * c / 1064e-9
    # Gamma_sc_1064 = 3*pi*c**2 / (2*hbar*w0**3) * (w/w0)**3 * (31e6 * 2 * pi)**2 *(1/(w0-w)+1/(w0+w))**2
    