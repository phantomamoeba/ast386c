#AST 386C Properties of Galaxies
#homework #3
#November 8,2018
#Dustin Davis

import sys

sys.path.append('../../elixer/')
import global_config as G
import spectrum as elixer_spectrum

# import specutils as su
# from astropy import units as u


import numpy as np
from scipy.integrate import quad
import math
import os.path as op
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.colors as colors
import matplotlib.cm as cmx
import astropy.table

import astropy.io.fits as fits
import astropy.modeling.functional_models as models
#import astropysics.models.SersicModel as sersic


#prominent lines in the rest-frame UV/optical/NIR
#(wavelength in angstroms)
# wave[AA] name
# 1215.67    Ly\alpha
# 1550.00	   CIV
# 1909	   CIII]
# 3727	   [OII]
# 4861.3	   H\beta
# 6549	   [NII]
# 6562.8	   H\alpha
# 6583	   [NII]
# 12818.	   Pa\beta
# 18751.	   Pa\alpha
#
#
# prominent lines in the rest-frame mm
# (frequency in MHz)
#
# # name freq[MHz]
# CO(1-0)	115271.2
# CO(2-1)	230538.
# CO(3-2)	345769.
# CO(4-3)	461040.8
# CI(1-0) 492000.
# CO(5-4)	576267.9
# CO(6-5)	691473.1
# CO(7-6)	806651.8
# CI(2-1) 809342.
# CO(8-7)	921799.7
# CO(9-8) 1037020.
# CO(10-9) 1152250.
# CO(11-10) 1267470.
# CO(12-11) 1373090.
# CO(13-12) 1478710.
# CO(14-13) 1584330.
# CII	  1898730.



###############################
#GLOBALS
###############################

BASEDIR = "../res"
OUTDIR = "out"
c_const = 2.99792458e18 #angstroms/sec


NIRSpectrum_AA = []
NIRSpectrum_cgs = [] #erg s^-1 cm^-2 AA^-1

MMSpectrum_GHz = []
MMSpectrum_Jy = []

##################################
#Support Functions
##################################

def read_NIRSpectrum():
    lam, Flam  = np.loadtxt(fname=op.join(BASEDIR,"nirspectrum_hw3.txt"),comments='#',dtype=float,unpack=True)

    interp_lam = np.arange(lam[0],lam[-1],2.0)
    interp_Flam = np.interp(interp_lam,lam,Flam)
    #return np.array(lam[1:-1]), np.array(Flam[1:-1]) #trim 1st and last, step size is different
    return np.array(interp_lam), np.array(interp_Flam) #trim 1st and last, step size is different

def read_MMSpectrum():
    freq, flux  = np.loadtxt(fname=op.join(BASEDIR,"mmspectrum_hw3.txt"),comments='#',dtype=float,unpack=True)
    return np.array(freq), np.array(flux)

def Fnu2Flam(freq,flux):
    return (flux * freq ** 2)/c_const

def Flam2Fnu(lam,flux):
    return (flux * lam ** 2)/c_const

def GHz2AA(GHz): #units are in GHz
    return c_const / (GHz * 1e9)

def AA2GHz(AA):
    return c_const / AA # c_const already in AA units

def Jy2cgs(Jy):
    return Jy * 1e-23



def prob1a():
    #plot both spectra
    #by eye, find peaks and ID for redshift
    #could fit gaussian to get mu
    #units of flux here do not really matter ... just looking for emission lines, don't care what units

    # lamb = NIRSpectrum_AA * u.AA
    # flux = NIRSpectrum_cgs * u.Unit('erg cm-2 s-1 AA-1')
    # spec = su.Spectrum1D(spectral_axis=lamb, flux=flux)
    # lines = plt.step(spec.spectral_axis, spec.flux)
    #
    #
    #

    #GAUSS_FIT_PIX_ERROR from 2.0 to 3.0
    #GAUSS_FIT_MAX_SIGMA from 10.0 to 17.0
    #...
    #if ( abs(fit_peak - raw_peak) > (raw_peak * 0.25) ):     #0.20 to 0.25

    #G.DEBUG_SHOW_GAUSS_PLOTS = True
    G.DISPLAY_ABSORPTION_LINES = False
    G.MAX_SCORE_ABSORPTION_LINES = 0.0  # the most an absorption line can contribute to the score (set to 0 to turn off)

    spec_obj = elixer_spectrum.Spectrum()  # todo: needs spec_obj.identifier and spec_obj.plot_dir

    spec_obj.set_spectra(NIRSpectrum_AA, NIRSpectrum_cgs*100.0,
                         errors=None, central=22789.1429788, values_units=1)  # ,fit_min_sigma=2.0)

 #   cw = spec_obj.find_central_wavelength()

    spec_obj.classify()  # might be none

    z = spec_obj.solutions[0].z

    plt.figure(figsize=(12,4))
    plt.title("Observed Flux density (NIR) z=%0.3f" %(z))
    plt.xlabel(r"Wavelength $\AA$")
    plt.ylabel("Flux [$erg s^{-1} cm^{-2} \AA^{-1}$]")
    plt.plot(NIRSpectrum_AA,NIRSpectrum_cgs)

    #todo: just plot the best fit lines (NaII and H_alpha) or (OIII and H_beta) 


    #red for H_alpha 6562.8
    plt.axvline(x=spec_obj.solutions[0].central_rest * (1+z), ls='dashed', c='r', zorder=1, alpha=0.5,label=r"$H_{\alpha}$")

    #NaII
    plt.axvline(x=6549.0*(1+z), ls='dashed', c='g', zorder=1, alpha=0.5, label=r"NaII")

    #other NaII 6583
    plt.axvline(x=6583.0*(1+z), ls='dashed', c='g', zorder=1, alpha=0.5,label=r"NaII")


    # for f in spec_obj.all_found_lines:  # this is an EmisssionLineInfo object
    #     plt.axvline(x=f.raw_x0, ls='dashed', c='k', zorder=1, alpha=0.5)

    # for s in spec_obj.solutions:
    #     print("z=%f, line=%s, score=%f, total lines=%d" %(s.z,s.name,s.score,len(s.lines)))
    #     for l in s.lines:
    #         if l.absorber:
    #             print ("   [absorber] line=%s rest=%f" %(l.name,l.w_rest))
    #         else:
    #             print("line=%s rest=%f" % (l.name, l.w_rest))

    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=0)
    plt.savefig(op.join(OUTDIR,"prob1a_1.png"))
    #plt.show()
    plt.close()

    plt.figure(figsize=(12, 4))
    plt.title("Observed Flux density (mm) z=%0.3f" %(z))
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Flux [mJy]")
    plt.plot(MMSpectrum_GHz*1000.0,MMSpectrum_Jy*1000.0)
    #CO(3-2)	345769.
    plt.axvline(x=345769.0 / (1 + z), ls='dashed', c='b', zorder=1, alpha=0.5, label="CO(3-2)")
    plt.legend(loc='upper right', bbox_to_anchor=(0.98, 0.98), borderaxespad=0)

    plt.savefig(op.join(OUTDIR,"prob1a_2.png"))
   # plt.show()
    plt.close()


def main():
    global  NIRSpectrum_AA , NIRSpectrum_cgs, MMSpectrum_GHz, MMSpectrum_Jy
    NIRSpectrum_AA,  NIRSpectrum_cgs = read_NIRSpectrum()
    MMSpectrum_GHz, MMSpectrum_Jy = read_MMSpectrum()

    G.logging.basicConfig(filename="hw3.log", level=G.LOG_LEVEL, filemode='w')
    plt.switch_backend('QT4Agg')

    #just a test
#    mm_AA = GHz2AA(MMSpectrum_GHz)
#    mm_cgs = Fnu2Flam(MMSpectrum_GHz*1e9,Jy2cgs(MMSpectrum_Jy))

    prob1a()



if __name__ == '__main__':
    main()
