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


BASEDIR = "../res"
OUTDIR = "out"


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


    exit(0)


if __name__ == '__main__':
    main()
