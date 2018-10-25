#AST 386C Properties of Galaxies
#homework #2
#October 16,2018
#Dustin Davis

import sys
sys.path.append('../hw1/')
import hw1_dd as hw1

# sys.path.append('../../elixer/')
# import science_image as si


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


def dist(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)

def read_calzetti():
    #return the interpolated wavelength array and A_lambda array
    # second column is  A in V-band as a function of wavelength;
    # s|t R_V == A_V/(A_B - A_V)  (or R_V = A_V/ E(B-V) ) where '_' == subscript
    # *** in other words, the second column is R_V*E(B-V)) ****
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


def chi_sqr(obs, exp, error=None):

    obs = np.array(obs)
    exp = np.array(exp)

    x = len(obs)

    if error is not None:
        error = np.array(error)

    if error is not None:
        c = np.sum((obs*exp)/(error*error)) / np.sum((exp*exp)/(error*error))
    else:
        c = 1.0

    chisqr = 0
    if error is None:
        error=np.zeros(np.shape(obs))
        error += 1.0

    for i in range(x):
            chisqr = chisqr + ((obs[i]-c*exp[i])**2)/(error[i]**2)
    return chisqr,c

def chi_sqr2D(obs, exp, error=None):

    obs = np.array(obs)
    exp = np.array(exp)

    x,y = np.shape(obs)

    if error is not None:
        error = np.array(error)

    if error is not None:
        c = np.sum((obs*exp)/(error*error)) / np.sum((exp*exp)/(error*error))
    else:
        c = 1.0

    chisqr = 0
    if error is None:
        error=np.zeros(np.shape(obs))
        error += 1.0

    for i in range(x):
        for j in range(y):
            chisqr = chisqr + ((obs[i][j]-c*exp[i][j])**2)/(error[i][j]**2)
    return chisqr,c


def getnearpos(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

#################################
# Commons
#################################
SPECTRA_GRID_AA, A_LAMBDA_GRID = read_calzetti()
SPECTRA_GRID_MICRONS = SPECTRA_GRID_AA / 10000.

SPECTRA_GRID_Hz = hw1.c_const / SPECTRA_GRID_AA
SPECTRA_GRID_Hz = np.flip(SPECTRA_GRID_Hz, axis=0) #flip so is in increasing frequencey order (reversed from wavelength)

EBV_0p1_GRID = 10 ** (-0.4 * A_LAMBDA_GRID * 0.1)
EBV_1p0_GRID = 10 ** (-0.4 * A_LAMBDA_GRID)

EBV_0p1_GRID_Hz = np.flip(EBV_0p1_GRID, axis=0) #since we flipped the frequence direction, have to do that here too
EBV_1p0_GRID_Hz = np.flip(EBV_1p0_GRID, axis=0)


def prob2a():
    #plot the attenuation (f_obs/f_intrinsic) = 10^(-0.4 A) for E(B-V) = 0.1 and E(B-V)= 1.0

    #A_v_0p1 = np.sum()
    start = getnearpos(SPECTRA_GRID_AA,4500)
    stop = getnearpos(SPECTRA_GRID_AA,7500)
    sum = 0
    for i in range(start,stop): #indicies of roughly V-Band (5060AA - 5940AA)
        sum += A_LAMBDA_GRID[i] * (SPECTRA_GRID_AA[i+1] - SPECTRA_GRID_AA[i])

    Av_1p0 = sum / (stop-start) #this is R_V*E(B-V) with E(B-V) == 1.0
    Av_0p1 = Av_1p0 * 0.1 #just trying to keep this clear

    #end up scaling by the same factor (0.1) on top and bottom, so Rv is the same
    #this is CORRECT ... for a given attenuation curve, Rv (over a wavelength range) is essentially constant
    #as any Change to E(B-V) is a change to A_B and A_V and the attenuation shifts up and down but the slope
    # does not change
    print("V-band [5000-7000AA]")
    print("R_V (for E(B-V) = 0.1) ~ %f" %(Av_0p1/0.1))
    print("R_V (for E(B-V) = 1.0) ~ %f" %(Av_1p0/1.0))


    plt.figure()
    plt.gca().set_xscale("linear")
    plt.gca().set_yscale("linear")
    plt.xlim(2.5,4.5)
    plt.ylim(-0.01, 1.01)
    plt.title("Fractional Attenuation by Wavelength")
    plt.ylabel(r'$f_{obs}/f_{int}$')  # / $L_{\odot}$) in  units')
    plt.xlabel(r'Log($\lambda$) [$\AA$]')

    plt.plot(np.log10(SPECTRA_GRID_AA), EBV_0p1_GRID, color='b',label="E(B-V)=0.1")
    plt.plot(np.log10(SPECTRA_GRID_AA), EBV_1p0_GRID, color='r',label="E(B-V)=1.0")

    rec = plt.Rectangle((np.log10(5000),0.0), np.log10(7000)-np.log10(5000),1.0, fill=True, lw=1,
                        color='g', alpha=0.5,label="Approx V-band")
    plt.gca().add_patch(rec)

    rec = plt.Rectangle((np.log10(4000),0.0), np.log10(5000)-np.log10(4000),1.0, fill=True, lw=1,
                        color='b', alpha=0.5,label="Approx B-band")
    plt.gca().add_patch(rec)

    plt.gca().tick_params(axis='y', which='minor', left='on')
    #lt.gca().set_xticks(list(ax.get_xticks()) + extraticks)
    plt.gca().set_yticks(np.arange(0.0,1.1,0.1))

    plt.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), borderaxespad=0)

    #plt.show()
    plt.savefig(op.join(OUTDIR,"hw2_prob2a.png"))


def prob2b(integrated_spectra_0,integrated_spectra_500,integrated_spectra_1000):


    #plot nine spectra age x extenction
    plt.figure()
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.xlim(1e2,28500)
    plt.ylim(1e-15,1000)
    plt.title("Integrated Population Spectra (Aged and Attenuated)")
    plt.ylabel(r'$\propto \nu L_\nu$')
    plt.xlabel(r'$\lambda$ [$\AA$]')

    lw = 1.0

    plt.plot(SPECTRA_GRID_AA, integrated_spectra_0, color='b',lw=lw,ls="solid")
    plt.plot(SPECTRA_GRID_AA, integrated_spectra_0*EBV_0p1_GRID, color='b',lw=lw,ls="--")
    plt.plot(SPECTRA_GRID_AA, integrated_spectra_0*EBV_1p0_GRID, color='b',lw=lw,ls=':')

    plt.plot(SPECTRA_GRID_AA, integrated_spectra_500, color='g',lw=lw,ls="solid")
    plt.plot(SPECTRA_GRID_AA, integrated_spectra_500 * EBV_0p1_GRID, color='g',lw=lw,ls="--")
    plt.plot(SPECTRA_GRID_AA, integrated_spectra_500 * EBV_1p0_GRID, color='g',lw=lw, ls=':')

    plt.plot(SPECTRA_GRID_AA, integrated_spectra_1000, color='r',lw=lw,ls="solid")
    plt.plot(SPECTRA_GRID_AA, integrated_spectra_1000 * EBV_0p1_GRID, color='r',lw=lw,ls="--")
    plt.plot(SPECTRA_GRID_AA, integrated_spectra_1000 * EBV_1p0_GRID, color='r', lw=lw,ls=':')

    blu_patch = mpatches.Patch(color='b', label='Aged 0 yr')
    grn_patch = mpatches.Patch(color='g', label='Aged 500 Myr')
    red_patch = mpatches.Patch(color='r', label='Aged 1 Gyr')

    solid = mlines.Line2D([], [], color='k', linestyle="solid", label='E(B-V) = 0.0')
    dash = mlines.Line2D([], [], color='k', linestyle="--", label='E(B-V) = 0.1')
    dots = mlines.Line2D([], [], color='k', linestyle=":", label='E(B-V) = 1.0')

    plt.legend(handles=[blu_patch,grn_patch,red_patch,solid,dash,dots],
               loc='lower right', bbox_to_anchor=(0.95, 0.05), borderaxespad=0)

    #plt.show()
    plt.savefig(op.join(OUTDIR,"hw2_prob2b.png"))
    plt.close()



def prob2c(integrated_spectra_0,integrated_spectra_500,integrated_spectra_1000):

    g_filter = hw1.Filter('g', op.join(BASEDIR, "subaru_g.txt"))
    r_filter = hw1.Filter('r', op.join(BASEDIR, "subaru_r.txt"))

    #take the spectra as is ... prop to wLw
    #divide by w to get Lw
    #convert to Lv (usual conversion)
    #convert wavelength grid to frequency
    #then get attenuation by frequency and use inegration(1/v * Lv * Tv *dv) / integration( (1/v * Tv *dv))

    # note: since integrated_spectra_0 alredy in wLw format, just multiple by w once more to get f*w**2 / c
    flux_freq_0 = np.flip(integrated_spectra_0 * SPECTRA_GRID_AA / hw1.c_const, axis=0)  # now prop to f_v
    flux_freq_500 = np.flip(integrated_spectra_500 * SPECTRA_GRID_AA / hw1.c_const, axis=0)  # now prop to f_v
    flux_freq_1000 = np.flip(integrated_spectra_1000 * SPECTRA_GRID_AA / hw1.c_const, axis=0)  # now prop to f_v


    Fg = g_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_0,False)
    Fr = r_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_0,False)
    g_r_color = -2.5*np.log10(Fg/Fr)
    print("Aged 0 yr, E(B-V) = 0 : g-r color = %0.4f" % g_r_color)


    Fg = g_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_0*EBV_0p1_GRID_Hz,False)
    Fr = r_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_0*EBV_0p1_GRID_Hz,False)
    g_r_color = -2.5*np.log10(Fg/Fr)
    print("Aged 0 yr, E(B-V) = 0 : g-r color = %0.4f" % g_r_color)

    Fg = g_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_0*EBV_1p0_GRID_Hz,False)
    Fr = r_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_0*EBV_1p0_GRID_Hz,False)
    g_r_color = -2.5*np.log10(Fg/Fr)
    print("Aged 0 yr, E(B-V) = 0 : g-r color = %0.4f" % g_r_color)



    Fg = g_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_500,False)
    Fr = r_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_500,False)
    g_r_color = -2.5*np.log10(Fg/Fr)
    print("Aged 500 Myr, E(B-V) = 0 : g-r color = %0.4f" % g_r_color)


    Fg = g_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_500*EBV_0p1_GRID_Hz,False)
    Fr = r_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_500*EBV_0p1_GRID_Hz,False)
    g_r_color = -2.5*np.log10(Fg/Fr)
    print("Aged 500 Myr, E(B-V) = 0 : g-r color = %0.4f" % g_r_color)

    Fg = g_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_500*EBV_1p0_GRID_Hz,False)
    Fr = r_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_500*EBV_1p0_GRID_Hz,False)
    g_r_color = -2.5*np.log10(Fg/Fr)
    print("Aged 500 Myr, E(B-V) = 0 : g-r color = %0.4f" % g_r_color)


    Fg = g_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_1000,False)
    Fr = r_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_1000,False)
    g_r_color = -2.5*np.log10(Fg/Fr)
    print("Aged 1 Gyr, E(B-V) = 0 : g-r color = %0.4f" % g_r_color)


    Fg = g_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_1000*EBV_0p1_GRID_Hz,False)
    Fr = r_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_1000*EBV_0p1_GRID_Hz,False)
    g_r_color = -2.5*np.log10(Fg/Fr)
    print("Aged 1 Gyr, E(B-V) = 0 : g-r color = %0.4f" % g_r_color)

    Fg = g_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_1000*EBV_1p0_GRID_Hz,False)
    Fr = r_filter.get_F_nu(SPECTRA_GRID_Hz, flux_freq_1000*EBV_1p0_GRID_Hz,False)
    g_r_color = -2.5*np.log10(Fg/Fr)
    print("Aged 1 Gyr, E(B-V) = 0 : g-r color = %0.4f" % g_r_color)




def prob2d(integrated_spectra_0,integrated_spectra_500,integrated_spectra_1000):

    filters = {'g': hw1.Filter('g', op.join(BASEDIR, "subaru_g.txt")),
               'r': hw1.Filter('r', op.join(BASEDIR, "subaru_r.txt")),
               'i': hw1.Filter('i', op.join(BASEDIR, "subaru_i.txt")),
               'z': hw1.Filter('z', op.join(BASEDIR, "subaru_z.txt")),
               'y': hw1.Filter('y', op.join(BASEDIR, "subaru_y.txt")),
               }

    myr = [0,500,1000]
    att = [np.ones(EBV_0p1_GRID_Hz.shape),EBV_0p1_GRID_Hz,EBV_1p0_GRID_Hz]

    flux_freq = {}
    flux_freq['0'] = np.flip(integrated_spectra_0 * SPECTRA_GRID_AA / hw1.c_const, axis=0)  # now prop to f_v
    flux_freq['500'] = np.flip(integrated_spectra_500 * SPECTRA_GRID_AA / hw1.c_const, axis=0)  # now prop to f_v
    flux_freq['1000'] = np.flip(integrated_spectra_1000 * SPECTRA_GRID_AA / hw1.c_const, axis=0)  # now prop to f_v

    #get flux in each filter and convert to proportional to uJy
    solutions = {}

    for age in myr:
        for i in range(len(att)):
            skey = str(age) + "myr_" + str(i)
            flux = {}
            for key in filters:
                flux[key] = filters[key].get_F_nu(SPECTRA_GRID_Hz, flux_freq[str(age)] * att[i],False)# * 1e29 #cgs to uJy

            flux['age'] = age
            flux['att'] = i
            solutions[skey] = flux

    #now we have (not scaled yet) 9 possible solutions

    # galaxy_photo.txt
    # wave_iso         fnu_uJy    fnu_err_uJy
    #      2330.00      3.69280     0.510179
    #      3300.00      10.4355      3.60430
    #   g  4500.00      45.0000      12.4340
    #  (v) 5270.00      56.4691      23.4045
    #   r  6580.00      52.2668      7.22092
    #   i  8140.00      57.6268      15.9229
    #   z  9500.00      53.4224      18.4515
    #   y  10500.0      66.2505      32.0350

    obs = np.array([45.0,52.2668,57.6268,53.4224,66.2505]) #grizy (no v)
    err = np.array([12.4340,7.22092,15.9229,18.4515,32.0350])
    wav = np.array([4500.,6580.,8140.,9500.,10500.])
    frq = hw1.c_const/wav

    all_obs = np.array([3.69280,10.4355,45.0,56.4691,52.2668,57.6268,53.4224,66.2505]) #grizy (no v)
    all_err = np.array([0.510179,3.60430,12.4340,23.4045,7.22092,15.9229,18.4515,32.0350])
    all_wav = np.array([2300.,3300.,4500.,5270.,6580.,8140.,9500.,10500.])
    all_frq = hw1.c_const/all_wav

    best_chisqr = 9e99
    best_sol = None
    best_scale = None

    # now find best fit
    for key in solutions:
        model = [ solutions[key]['g'],
                  solutions[key]['r'],
                  solutions[key]['i'],
                  solutions[key]['z'],
                  solutions[key]['y'] ]

        #model = [v for v in solutions[key].values()] #in grizy order ... not guaranteed (could use OrderedDict though)

        chi2, scale = chi_sqr(obs, model, err)
        solutions[key]['chi2'] = chi2
        solutions[key]['scale'] = scale

        print(key, chi2)

        if chi2 < best_chisqr:
            best_chisqr = chi2
            best_sol = key
            best_scale = scale

    print ("*** Best", best_sol,best_chisqr)

    plt.figure()
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")

#    plt.xlim(2000, 100)
    plt.xlim(1000,20000)
    plt.ylim(1, 200)
    plt.title(r"SED Best Fit ($\chi^2$)")
    plt.ylabel(r'$f_{\nu}$ [$\mu Jy$]')
    #plt.xlabel(r'$\nu$ [tera-Hz]')
    plt.xlabel(r'$\lambda$ [$\AA$]')
    lw = 1.0

    for key in solutions.keys():
        scale = solutions[key]['scale']
        sed = flux_freq[str(solutions[key]['age'])]
        atten = att[solutions[key]['att']]

        #plt.plot(SPECTRA_GRID_Hz /1e12, sed * atten * scale, color='k', lw=lw, ls=":", alpha=0.5)
        plt.plot(SPECTRA_GRID_AA, np.flip(sed * atten * scale,axis=0), color='k', lw=lw, ls=":", alpha=0.5)

    #overplot the best
    scale = solutions[best_sol]['scale']
    sed = flux_freq[str(solutions[best_sol]['age'])]
    atten = att[solutions[best_sol]['att']]


  # plt.plot(SPECTRA_GRID_Hz/1e12,sed*atten*scale, color='b', lw=lw, ls="solid")
    plt.plot(SPECTRA_GRID_AA, np.flip(sed * atten * scale,axis=0), color='b', lw=lw, ls="solid")

    #plt.errorbar(all_frq/1e12,all_obs,yerr=all_err,color='r',fmt='o')
    plt.errorbar(all_wav, all_obs, yerr=all_err, color='r', fmt='o')

    plt.savefig(op.join(OUTDIR,"hw2_prob2d.png"))
    #plt.show()
    plt.close()



def prob3a(x1,y1,x2,y2):
    #todo: code it up here rather than just use calculator
    #todo: see if there is a python extension that tracks errors rather than use definition |dQ| ...
    pass



def prob3b():
    #image is already cropped, so no need for a true cutout (just use the whole data payload)
    data = fits.getdata("../res/ngc5055.fits")
    pa = 100.0 #position angle
    max_count_core = 23755
    #ellipticity = (425.8-202.6)/425.8  #roughly 61 deg

    ellipticity = (425.44 - 279.89) / 425.44 #roughly 49 deg


    h_init = int(dist(450.,450.,295.,410.)) #using contours on ds9 image, approx location of center counts to 1/e
    #this is the center (450,450) to one edge(295,410)
    best_h = 259.63#308.92 #309. #half-light radius ... r_1/2 for r_e = r_1/2 / ln(2) ~~ 445.68 pixels

    x,y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0])) #in ds9, y is 1st index, x is second


    #minimize the residual to fit h?

    #
    # d = data[288:594,168:640]
    # max_count_core = np.max(d)
    # best_h = h_init
    # best_chi2 = 9e99
    # for h in range(h_init+90,h_init+120):
    #     model = models.Sersic2D(amplitude=1, r_eff=h,n=1.0,
    #                             x_0=len(x)/2.,y_0=len(y)/2.,
    #                             ellip=ellipticity,theta=(pa-90.)*np.pi/180.)
    #     img = model(x, y)[288:594,168:640]
    #     img *= max_count_core/np.max(img)
    #
    #     chi2, scale = chi_sqr2D(d, img, None)
    #     print (chi2,h)
    #
    #     if chi2 < best_chi2:
    #         best_chi2 = chi2
    #         best_h = h
    #
    # print("Best",best_chi2,best_h)

    #amplitude = surface brightness at r_eff not at center, but still okay, since scaling up afterward so the core
    # counts match

    #refine from 309
    # h_init = 309.
    # h_init = 260.
    # d = data[288:594,168:640]
    # max_count_core = np.max(d)
    # best_h = h_init
    # best_chi2 = 9e99
    # for deci in range(-100,101):
    #     h = float(h_init)+float(deci)/100.
    #     model = models.Sersic2D(amplitude=1, r_eff=h,n=1.0,
    #                             x_0=len(x)/2.,y_0=len(y)/2.,
    #                             ellip=ellipticity,theta=(pa-90.)*np.pi/180.)
    #     img = model(x, y)[288:594,168:640]
    #     img *= max_count_core/np.max(img)
    #
    #     chi2, scale = chi_sqr2D(d, img, None)
    #     print (chi2,h)
    #
    #     if chi2 < best_chi2:
    #         best_chi2 = chi2
    #         best_h = h
    #
    # print("Best",best_chi2,best_h)

    #best_h = 259.63

    # # fit scale
    #d = data[288:594,168:640]
    # max_count_core = np.max(d)
    # best_h = h_init
    # best_chi2 = 9e99
    # best_scale = 4459
    # for sc in range(best_scale-100,best_scale+100):
    #     model = models.Sersic2D(amplitude=sc, r_eff=best_h,n=1.0,
    #                             x_0=len(x)/2.,y_0=len(y)/2.,
    #                             ellip=ellipticity,theta=(pa-90.)*np.pi/180.)
    #     img = model(x, y)[288:594,168:640]
    #
    #     chi2, scale = chi_sqr2D(d, img, None)
    #     print ("Scale", chi2,sc)
    #
    #     if chi2 < best_chi2:
    #         best_chi2 = chi2
    #         best_scale = sc
    #
    # print("Best Scale",best_chi2,best_scale)



    #try scaling
    # best_h = h_init
    # best_chi2 = 9e99
    # best_scale = 4459
    # model = models.Sersic2D(amplitude=1, r_eff=best_h, n=1.0,
    #                              x_0=len(x)/2.,y_0=len(y)/2.,
    #                              ellip=ellipticity,theta=(pa-90.)*np.pi/180.)
    # for sc in range(best_scale-100,best_scale+100):
    #     img = model(x, y)  # .swapaxes(0,1)
    #     img *= sc
    #     chi2, scale = chi_sqr2D(d, img, None)
    #     print ("Scale", chi2,sc)


    model = models.Sersic2D(amplitude=1, r_eff=best_h, n=1.0,
                            x_0=len(x) / 2., y_0=len(y) / 2.,
                            ellip=ellipticity, theta=(pa - 90.) * np.pi / 180.)
    img = model(x, y)  # .swapaxes(0,1)
    img *= max_count_core / np.max(img)


    plt.figure(figsize=(13, 4)) # 2 cols, 1row

    plt.subplot(121) #image
    plt.title("NGC 5055")
    plt.imshow(data, origin="lower",cmap="gray")
    cbar = plt.colorbar()
    cbar.set_label('Counts')
    plt.xlabel("pix")
    plt.ylabel("pix")


    plt.subplot(122) #model
    plt.title(r"$\Sigma (r)$ model")
    plt.imshow(img, origin="lower", cmap="gray")
    plt.xlabel("pix")
    plt.ylabel("pix")

    cbar = plt.colorbar()
    cbar.set_label('Counts')

    plt.savefig(op.join(OUTDIR,"hw2_prob3b.png"))
    plt.show()
    return data, model, img


def prob3c(data,model,image):

    residual = data - image

    plt.figure(figsize=(16, 4))  # 2 cols, 1row

    plt.subplot(121)
    plt.title("Original Image (cropped)")
    plt.imshow(data, origin="lower", cmap="gray")

    cbar = plt.colorbar()
    cbar.set_label('Counts')


    plt.subplot(122)
    plt.title("Residual Image (cropped)")
    plt.imshow(residual, origin="lower", cmap="gray")

    cbar = plt.colorbar()
    cbar.set_label('Counts')


    plt.savefig(op.join(OUTDIR, "hw2_prob3c1.png"))
    plt.show()
    plt.close()


    plt.figure()
    plt.title("Residual Image (histogram)")
    #plt.gca().yaxis.set_label_position("right")
    n,b,_ = plt.hist(residual.flatten(),bins=1000)#,range=[0,1.0]))
    print("Median at: ", np.median(n))
    plt.xlabel("Residual Counts")
    plt.ylabel("Number of Pixels")
    plt.xlim(xmin=-6000,xmax=6000)
    plt.savefig(op.join(OUTDIR, "hw2_prob3c2.png"))
    plt.show()
    plt.close()



def prob3e():
    from scipy.optimize import curve_fit

    def exponential(r, h, sigma_cent):
        return sigma_cent * np.exp(-r / h)

    #sigma_r = [20.1,21.1,22.7,25.25] #4 values
    #r = [1.,2.,4.,8.] # values

    sigma_r = [331.13,131.83,30.20,2.88] #4 values
    r = [2730.,5470.,10940.,21870.] # values

    parm, pcov = curve_fit(exponential, r, sigma_r, p0=[3430,600])

    print (parm)

    plt.figure()
    x = np.arange(1, 100000.,100)
    y = exponential(x,parm[0],parm[1])
    y2 = exponential(x,3430,600)

    plt.plot(x,y)
    plt.plot(x,y2)
    plt.scatter(r,sigma_r)
    plt.show()
    plt.close()






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

        #now, put these on the same grid as SPECTRA_GRID_AA ... same step and start, so just truncate
        integrated_spectra_0 = integrated_spectra_0[:len(SPECTRA_GRID_AA)]
        integrated_spectra_500 = integrated_spectra_500[:len(SPECTRA_GRID_AA)]
        integrated_spectra_1000 = integrated_spectra_1000[:len(SPECTRA_GRID_AA)]

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
    #prob2b(integrated_spectra_0,integrated_spectra_500,integrated_spectra_1000)
    #prob2c(integrated_spectra_0, integrated_spectra_500, integrated_spectra_1000)
    #prob2d(integrated_spectra_0, integrated_spectra_500, integrated_spectra_1000)

    # data, model, image = prob3b()
    # prob3c(data[288:594,168:640],model,image[288:594,168:640])

    prob3e()

    exit(0)


if __name__ == '__main__':
    main()
