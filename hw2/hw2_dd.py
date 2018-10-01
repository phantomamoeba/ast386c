#AST 386C Properties of Galaxies
#homework #2
#October 16,2018
#Dustin Davis

import sys
sys.path.append('../hw1/')
import hw1_dd as hw1


import numpy as np
from scipy.integrate import quad
import math
import os.path as op
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import astropy.table


###############################
#GLOBALS
###############################

BASEDIR = "../res"
OUTDIR = "out"

MIN_MASS = 0.08  #m_sun 0.0076
MAX_MASS = 100.0  #m_sun 0.0076
MIN_WAVE = 90.0 #AA
MAX_WAVE = 1e6 #AA



##################################
#Support Functions
##################################

def read_calzetti():
    #return the interpolated wavelength array and A_lambda array
    #fo_fi_grid is f_observered/f_intrinsic for the 1AA width spectra grid and E(B-V)==1.0
    w,a = np.genfromtxt(op.join(BASEDIR,"calzetti01.txt"),unpack=True)
    w = np.insert(w,0,90.0)
    a = np.insert(a,0,90.5)

    #per Characterizing Dust Attenuation in Local Star-forming Galaxies: Near-infrared Reddening and Normalization
    # Battisti, Calzetti, Chary, 2017
    w = np.append(w,28500)
    a = np.append(a,0)
    spectra_grid = np.arange(MIN_WAVE, MAX_WAVE, 1)

    a_interp = np.interp(spectra_grid, w, a)

    return spectra_grid,a_interp



def chisqr(obs, exp, error=None):
    chisqr = 0
    if error is None:
        error=np.zeros(len(obs))
        error += 1.0

    for i in range(len(obs)):
        chisqr = chisqr + ((obs[i]-exp[i])**2)/(error[i]**2)
    return chisqr



#################################
# Commons
#################################
SPECTRA_GRID_AA, A_LAMBDA_GRID = read_calzetti()
SPECTRA_GRID_MICRONS = SPECTRA_GRID_AA / 10000.


def prob2a():
    #plot the attenuation (f_obs/f_intrinsic) = 10^(-0.4 A) for E(B-V) = 0.1 and E(B-V)= 1.0
    #don't need an IMF, this is just vs wavelength

    e0p1_grid = 10 ** (-0.4 * A_LAMBDA_GRID * 0.1)
    e1p0_grid = 10 ** (-0.4 * A_LAMBDA_GRID)

    plt.figure()
    plt.gca().set_xscale("linear")
    plt.gca().set_yscale("linear")
    plt.xlim(2.5,4.5)
    plt.ylim(-0.01, 1.01)
    plt.title("Fractional Attenuation by Wavelength")
    plt.ylabel(r'$f_{obs}/f_{int}$')  # / $L_{\odot}$) in  units')
    plt.xlabel(r'Log($\lambda$) [$\AA$]')

    leg_1, = plt.plot(np.log10(SPECTRA_GRID_AA), e0p1_grid, color='b')
    leg_2, = plt.plot(np.log10(SPECTRA_GRID_AA), e1p0_grid, color='r')


    plt.legend([leg_1, leg_2],
               ('E(B-V) = 0.1','E(B-V) = 1.0'),
               loc='upper left', bbox_to_anchor=(0.02, 0.98), borderaxespad=0)

    #plt.show()
    plt.savefig(op.join(OUTDIR,"hw2_prob2a.png"))



def main():


    ## from hw1 ... run once to get the spectra as files, then reload files there after
    #prob_4a
    #
    # #need these for most of what follows, so just do it once
    if False:
        mass_step = 0.01
        mass_grid, n_pdf, m_pdf = hw1.do_salpeter(0.08, 100.0, mass_step)

        Mass, Teff, Lum, mx, tx, lx = hw1.read_EEM_file( mass_grid)  # (min_mass,max_mass,step) #m,t,l are the original few data
        wavelength_grid = np.arange(90, 1.6e6, 1)  # in angstroms (wavelengths_grid)
        #
        #hw1.prob_4b(mass_grid=mass_grid, n_pdf=n_pdf, Teff=Teff, wavelength_grid=wavelength_grid)
        hw1.prob_4c(mass_grid=mass_grid, n_pdf=n_pdf, Teff=Teff, wavelength_grid=wavelength_grid,age=0)
        #
        # #have to rebuild for different ages (remaining masses)
        mass_grid, n_pdf, m_pdf = hw1.do_salpeter(0.08, 2.82, mass_step)
        Mass, Teff, Lum, mx, tx, lx = hw1.read_EEM_file(mass_grid)
        hw1.prob_4d1(mass_grid=mass_grid, n_pdf=n_pdf, Teff=Teff, wavelength_grid=wavelength_grid,age=500)
        #
        # #have to rebuild for different ages (remaining masses)
        mass_grid, n_pdf, m_pdf = hw1.do_salpeter(0.08, 2.14, mass_step)
        Mass, Teff, Lum, mx, tx, lx = hw1.read_EEM_file(mass_grid)
        hw1.prob_4d2(mass_grid=mass_grid, n_pdf=n_pdf, Teff=Teff, wavelength_grid=wavelength_grid,age=1000)
    else:

        wavelength_grid = np.load(op.join(OUTDIR,"spec_wave.npy"),allow_pickle=True)

        spec_0 = { 'M': np.load(op.join(OUTDIR,"spec_m0p08_m0p43_age0.npy"),allow_pickle=True),
                 'FGK': np.load(op.join(OUTDIR,"spec_m0p43_m2_age0.npy"),allow_pickle=True),
                  'BA': np.load(op.join(OUTDIR,"spec_m2_m20_age0.npy"),allow_pickle=True),
                   'O': np.load(op.join(OUTDIR,"spec_m20_m100_age0.npy"),allow_pickle=True)}

        spec_500 = {'M': np.load(op.join(OUTDIR, "spec_m0p08_m0p43_age500.npy"), allow_pickle=True),
                  'FGK': np.load(op.join(OUTDIR, "spec_m0p43_m2_age500.npy"), allow_pickle=True),
                  'BA': np.load(op.join(OUTDIR, "spec_m2_m20_age500.npy"), allow_pickle=True),
                  'O': np.load(op.join(OUTDIR, "spec_m20_m100_age500.npy"), allow_pickle=True)}

        spec_1000 = {'M': np.load(op.join(OUTDIR, "spec_m0p08_m0p43_age1000.npy"), allow_pickle=True),
                  'FGK': np.load(op.join(OUTDIR, "spec_m0p43_m2_age1000.npy"), allow_pickle=True),
                  'BA': np.load(op.join(OUTDIR, "spec_m2_m20_age1000.npy"), allow_pickle=True),
                  'O': np.load(op.join(OUTDIR, "spec_m20_m100_age1000.npy"), allow_pickle=True)}

        integrated_spectra_0 = spec_0['M'] + spec_0['FGK'] + spec_0['BA'] + spec_0['O']
        integrated_spectra_500 = spec_500['M'] + spec_500['FGK'] + spec_500['BA'] + spec_500['O']
        integrated_spectra_1000 = spec_1000['M'] + spec_1000['FGK'] + spec_1000['BA'] + spec_1000['O']

        # #test
        # fig = plt.figure(figsize=(9, 6))
        # plt.gca().set_xscale("log")
        # plt.gca().set_yscale("log")
        # plt.xlim(2e2, 3e4)  # out to about 3 microns
        # plt.ylim(1e-3, 1e3)
        # plt.ylabel(r'$\propto \nu L_{\nu}$')  # $[erg\ s^{-1}\ cm^{-2}]$')
        # plt.xlabel(r'$\lambda$ [$\AA$]')
        # #
        # spec = spec_1000
        # #
        # plt.plot(wavelength_grid, spec['M'], color='red')  # lightest
        # plt.plot(wavelength_grid, spec['FGK'], color='orange')
        # plt.plot(wavelength_grid, spec['BA'], color='green')
        # plt.plot(wavelength_grid, spec['O'], color='blue')  # heaviest
        #
        # plt.plot(wavelength_grid, integrated_spectra_0, color='blue',ls=":")  # lightest
        # plt.plot(wavelength_grid, integrated_spectra_500, color='green',ls=":")
        # plt.plot(wavelength_grid, integrated_spectra_1000, color='red',ls=":")
        #
        # fig.tight_layout()
        # plt.show()

    prob2a()
    print("***** Todo: 2(a) what R_v values do these correspond to....?")

    exit(0)


if __name__ == '__main__':
    main()
