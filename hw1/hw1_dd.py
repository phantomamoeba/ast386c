#AST 386C Properties of Galaxies
#homework #1
#September 25,2018
#Dustin Davis


import numpy as np
from scipy.integrate import quad
import math
import os.path as op
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import astropy.table


BASEDIR = "res"
OUTDIR = "out"

c_const = 2.99792458e18 #angstroms/sec
pc_cm = 3.086e18
L_sun = 3.839e33
M_sun = 1.9894*10**33 #grams
G_const = 6.67259*10**(-8) #cm^3 g^-1 s^-2

MIN_MASS = 0.08  #m_sun 0.0076

def jansky_to_erg(j):
    return j * 1.0e-23

def Fw2Fv(Fw,w): #lambda to nu (wavelength to frequency); c in angstroms/sec
    return Fw*(w**2)/c_const

def Fv2Fw(Fv,v): #
    return Fv*(v**2)/c_const


class Filter:
    def __init__(self,name,file):
        self.name = name
        self.w = None
        self.Tw = None

        self.v = None
        self.Tv = None

        self.read(file)

    def read(self,file):
        self.w, self.Tw = np.loadtxt(file, unpack=True)  # wavelength and effective transmission efficiency

        self.v = c_const/self.w
        self.Tv = self.Tw #roughly okay

    #transmission efficiency
    def getTw(self,wavelength):

        if (wavelength < (self.w[0]-(self.w[1]-self.w[0]))) or (wavelength > (self.w[-1] + (self.w[-1] - self.w[-2]))):
            return 0.0

        #interpolate the Tw
        return np.interp(wavelength,self.w,self.Tw)

    def getTv(self,frequency):

        if (frequency > (self.v[0]-(self.v[1]-self.v[0]))) or (frequency < (self.v[-1] + (self.v[-1] - self.v[-2]))):
            return 0.0

        #interpolate the Tw
        return np.interp(frequency,self.v,self.Tv)

    def get_range(self):
        #some of the first and last are zero, but it is close enough (the next adjacent is always non-zero)
        return self.w[0],self.w[-1]




class Star:
    def __init__(self,classification,file,distance=7.68):
        self.cls = classification
        self.distance = distance #in pc
        self.w = None
        self.Sv = None

        self.v = None
        self.Sw = None

        self.vLv = None
        self.wLw = None

        self.read(file)

    def read(self,file):
        self.w, self.Sv = np.loadtxt(file, unpack=True)  # wavelength and flux

        self.Sw = self.Sv*c_const/(self.w**2)
        self.v = c_const/self.w

        self.vLv = self.v*self.Sv*4*math.pi*(self.distance*pc_cm)**2
        self.wLw = self.w*self.Sw*4*math.pi*(self.distance*pc_cm)**2

    def get_average_flux(self,filter):
        total_f = 0.0
        count = 0
        wb,wr = filter.get_range()
        for w,f in zip(self.w,self.Sv):
            if wb <= w <= wr:
                total_f += f
                count += 1

        if count > 0:
            return total_f/count
        else:
            return 0.0

    def get_ab_mag(self,filter):
        #since this is discrete, just run two sums rather than two integrals
        #and use the (variable) difference to the next list element as dv (we lose the last one, but for tihs
        #purpose it won't matter)
        top = 0.
        bot = 0.
        #using Tw instead of Tv, but the indexing is the same for v and w in the source (since translated v as c/w)
        #so the Tv[v] == Tw[w] for the v that matches the w
        #since Jy is a flux density (in frequency) "integrate" (here, sum up) in flux density (frequency)
        for i in range(len(self.v)-1):
             top += 1./self.v[i] * self.Sv[i]*filter.getTw(self.w[i])*(self.v[i+1]-self.v[i])
             bot += 1./self.v[i] * filter.getTw(self.w[i]) * (self.v[i + 1] - self.v[i])

        fflux_v = top / bot

        m_ab = -2.5 * math.log((fflux_v/(jansky_to_erg(3631.0))),10)
        return m_ab


    def get_abs_ab_mag(self,filter,d):
        m = self.get_ab_mag(filter)
        M = 5.0 -5.0 * math.log(d,10) + m
        return M



def prob_2a(filters,star):
    flux_scale = 2.0e19
    wave_scale = 1.0

    plt.figure(figsize=(8,5))
    plt.title("Zoomed A0V Spectrum and Filters")
    plt.xlabel(r'$\lambda$ [$\AA$]')
    plt.ylabel(r'$S_{\nu}$ [%g $erg\ s^{-1}\ cm^{-2}\ \AA^{-1}$]' 
               '\nand Transmission Efficiency' %flux_scale)

    plt.xlim((min(filters['g'].w)-1000,max(filters['y'].w)+1000))

    jet = plt.get_cmap('jet')
    cNorm = colors.Normalize(vmin=0, vmax=7)  # O to M
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    color = 1

    plt.plot(star.w*wave_scale, star.Sv*flux_scale, color=scalarMap.to_rgba(0),label="A0V")
    plt.axhline(y=jansky_to_erg(3631.0)*flux_scale, color=scalarMap.to_rgba(color), ls='--',label="3631 Jy")

    for key in filters:
        color += 1
        plt.plot(filters[key].w, filters[key].Tw, color=scalarMap.to_rgba(color),label="%s-band"%key)
        plt.text(np.median(filters[key].w),0.2,key,color=scalarMap.to_rgba(color))

    plt.legend(loc="upper left",bbox_to_anchor=(1.05, 1.0))

    plt.tight_layout()

    plt.savefig(op.join(OUTDIR,"hw1_p2a.png"))



def prob_2b(filters,star):
    distance = 7.68
    for key in filters:
        mag = star.get_ab_mag(filters[key])
        Mag = star.get_abs_ab_mag(filters[key], distance)
        print("Vega %s-band mag_ab, Mag_ab = %f,%f" % (key,mag, Mag))

def prob_2d(star):

    fig = plt.figure()
    plt.title('Vega Luminosity')
    plt.ylabel(r'$\nu L_\nu$ & $\lambda L_\lambda$ [$erg\ s^{-1}$]')
    plt.xlabel('Wavelength [$\AA$]')

    plt.yscale('log')
    plt.xscale('log')

    plt.ylim(1e29,1e36)
    plt.xlim(1e3, 5e4)
    # plt.xlim(xmin, xmax)
    plt.plot(star.w, star.vLv, color='b',linewidth=1,ls="-",label=r'$\nu L_\nu$')
    plt.plot(star.w, star.wLw, color='g',linewidth=4,alpha=0.4,label=r'$\lambda L_\lambda$')
    plt.axhline(y=L_sun,color="r",lw=1,ls=":",label=r'$L_{\odot} (bol)$')

    peak_idx = np.argmax(star.vLv)
    mul = star.vLv[peak_idx]/L_sun
    #plt.axvline(x=star.w[peak_idx],color="g",label=r"Peak Flux (%0.1f x $L_{\odot}$)" % mul)
    plt.scatter(x=star.w[peak_idx],y=star.vLv[peak_idx], marker='o',s=150,edgecolors='r',facecolors='none',
                label=r"Peak Lum (%0.1f x $L_{\odot}$)" % mul)

    plt.legend(loc='lower right', bbox_to_anchor=(0.9, 0.1), borderaxespad=0)

    fig.tight_layout()
    #plt.show()
    plt.savefig(op.join(OUTDIR, "hw1_p2d.png"))

    fig = plt.figure()
    plt.title('Vega Luminosity (in Solar Units)')
    plt.ylabel(r'$\nu L_\nu$ & $\lambda L_\lambda$ [$L_{\odot}$]')
    plt.xlabel('Wavelength [$\AA$]')

    plt.yscale('log')
    plt.xscale('log')

    # plt.xlim(xmin, xmax)
    plt.plot(star.w, star.vLv/L_sun, color='b', linewidth=1, ls="-", label=r'$\nu L_\nu$')
    plt.plot(star.w, star.wLw/L_sun, color='g', linewidth=4, alpha=0.4, label=r'$\lambda L_\lambda$')

    peak_idx = np.argmax(star.vLv)
    mul = star.vLv[peak_idx] / L_sun
    # plt.axvline(x=star.w[peak_idx],color="g",label=r"Peak Flux (%0.1f x $L_{\odot}$)" % mul)
    plt.scatter(x=star.w[peak_idx], y=star.vLv[peak_idx]/L_sun, marker='o', s=150, edgecolors='r', facecolors='none',
                label=r"Peak Lum (%0.1f x $L_{\odot}$)" % mul)

    plt.ylim(1, 100)
    plt.xlim(1e3, 5e4)

    plt.legend(loc='upper right')#, bbox_to_anchor=(0.9, 0.1), borderaxespad=0)

    fig.tight_layout()
    # plt.show()
    plt.savefig(op.join(OUTDIR, "hw1_p2d-2.png"))



# 1/ln(10) ~ 0.434
# -0.434/1.35 ~ -0.32148
#-0.434/0.35 ~ -1.24
# though, really, these are all in relative fractions so the leading 1/ln(10) * 1/1.35 = -0.434/1.35 does not really matter ...
#this is result of the definite integral between m_low to m_high of m^-2.35 dm
#this is not a TRUE number of stars ... it is a fractional representation
def salpeter_num_stars(m_low,m_high):
    return -0.32148*(m_high**(-1.35)-m_low**(-1.35))

#this is result of the definite integral between m_low to m_high of m*m^-2.35 dm
def salpeter_mass_stars(m_low, m_high):
    return -1.24*(m_high**(-0.35)-m_low**(-0.35))


#this is the pdf N(m1,m2) the fractional number of stars between m1 and m2
def salpeter_stepped_numbers(m_low,m_high,step=0.01):
    m = []
    total_number = salpeter_num_stars(m_low, m_high)
    mass_range = np.arange(m_low,m_high+step,step)
    for i in mass_range:
        m.append(salpeter_num_stars(i,i+step))
    return np.array(m)/total_number

#this is the mass pdf M(m1,m2) (fractional mass between m1 and m2)
def salpeter_stepped_masses(m_low,m_high,step=0.01):
    m = []
    total_mass = salpeter_mass_stars(m_low, m_high)
    mass_range = np.arange(m_low, m_high + step, step)
    for i in mass_range:
        m.append(salpeter_mass_stars(i,i+step))
    return np.array(m)/total_mass


def do_salpeter(m_low,m_high,stepsize=0.01):
    mass_grid = np.arange(m_low,m_high+stepsize,stepsize)
    numbers_pdf = salpeter_stepped_numbers(m_low,m_high,stepsize)
    masses_pdf = salpeter_stepped_masses(m_low,m_high,stepsize)

    return mass_grid, numbers_pdf, masses_pdf



def salpeter_log_space_masses(logM_low, logM_high):
    m = []
    total_mass = salpeter_num_stars(10**logM_low, 10**logM_high)

    mass_range = np.logspace(logM_low, logM_high, 1000)
    for i in range(len(mass_range) - 1):
        m.append(salpeter_mass_stars(10 ** mass_range[i], 10 ** (mass_range[i + 1])))
        m.append(salpeter_mass_stars(10 ** mass_range[-1], 10 ** (mass_range[-1] + (mass_range[-1] - mass_range[-2]))))

    return np.array(m)/total_mass

def salpeter_log_space_numbers(logM_low, logM_high):
    m = []
    total_number = salpeter_num_stars(10**logM_low, 10**logM_high)

    mass_range = np.logspace(logM_low, logM_high,1000)
    for i in range(len(mass_range)-1):
        m.append(salpeter_num_stars(10**mass_range[i],10**(mass_range[i+1])))
    #last one
    m.append(salpeter_num_stars(10 ** mass_range[-1], 10 ** (mass_range[-1]+(mass_range[-1]-mass_range[-2]))))

    return np.array(m)/total_number

def salpeter_log_space(m_low, m_high):
    logM_low = np.log10(m_low)
    logM_high = np.log10(m_high)
    mass_grid = np.logspace(logM_low, logM_high,1000)
    numbers_pdf = salpeter_log_space_numbers(logM_low, logM_high)
    masses_pdf = salpeter_log_space_masses(logM_low, logM_high)
    return mass_grid, numbers_pdf, masses_pdf




def prob_3a():

    min_mass = 0.08
    max_mass = 100.0

    mass_grid,n_pdf,m_pdf = do_salpeter(min_mass,max_mass,0.01)

    exp_mass = np.sum(mass_grid*n_pdf)
    cdf = np.flip(np.cumsum(np.flip(n_pdf,0)),0)


    plt.figure()
    plt.gca().set_xscale("log")
    plt.xlim(min_mass,max_mass)
    plt.ylim(0.0,1.01)
    plt.title(r"Salpeter IMF ($\alpha$ = -2.35)")
    plt.xlabel("$M_{*}/M_{\odot}$")
    plt.ylabel("Cumulative Mass Fraction: f(>m)")

    plt.plot(mass_grid,cdf)

    val = min(mass_grid, key=lambda mass: abs(exp_mass-mass))
    exp_mass_y = cdf[np.where(mass_grid == val)]
    plt.scatter(exp_mass,exp_mass_y,color='r',marker='+',linewidth=1,s=100)#facecolors='none'
    plt.annotate(
        r"<m> $\approx$ %0.2f $M_{\odot}$" % exp_mass,
        xy=(exp_mass,exp_mass_y), xytext=(100, 40),
        textcoords='offset points', ha='right', va='bottom',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    #plt.show()
    plt.savefig(op.join(OUTDIR, "hw1_p3a.png"))


def luminosity(mass):
    #from 3b problem
    if mass < 0.43:
        return 0.23*(mass)**2.3
    elif mass < 2.0:
        return mass**4.0
    elif mass < 20.0:
        return 1.5*mass**3.5
    else:
        return 3200.0*mass

def prob_3d():

    plt.close('all')



    min_mass = 0.08
    max_mass = 100.0
    mass_step = 0.01

    mass_grid,n_pdf,m_pdf = do_salpeter(min_mass,max_mass,mass_step)

    #mass_grid, n_pdf, m_pdf = salpeter_log_space(min_mass, max_mass)

    lum_grid = []
    for m in mass_grid:
        lum_grid.append(luminosity(m))

    lum_grid = np.array(lum_grid)

    weighted_lums = n_pdf*lum_grid

    #normalize
    weighted_lums = weighted_lums/np.sum(weighted_lums)

    #want 100% at 0.08 mass and 0% at 100.0 mass
    cdf = np.flip(np.cumsum(np.flip(weighted_lums,0)),0)

    plt.figure(figsize=(9, 6))
    plt.gca().set_xscale("log")
    plt.xlim(min_mass,max_mass)
    #plt.ylim(0.0,1.0)
    plt.title(r"Salpeter IMF ($\alpha$ = -2.35)")
    plt.ylabel("Cumulative Luminosity Fraction $f_L$(>m)")
    plt.xlabel("$M_{*}/M_{\odot}$")
    plt.plot(mass_grid,cdf,lw=10., alpha=0.4, c='k',label="Direct Sum")


    #analytically ... piecewise integration
    mg20 = np.linspace(20.,100.,100)
    mg2 = np.linspace(2.,19.999, 100)
    mg43 = np.linspace(0.43,1.999,100)
    mg08 = np.linspace(0.08,0.42999,100)

    # mg20 = []
    # mg2 = []
    # mg43 = []
    # mg08 = []
    #
    # for m in np.logspace(-1.0969,2.,1000):
    #     mass = 10.**m
    #     if mass < 0.43:
    #         mg08.append(mass)
    #     elif mass < 2.0:
    #         mg43.append(mass)
    #     elif mass < 20.0:
    #         mg2.append(mass)
    #     else:
    #         mg20.append(mass)
    #
    # mg20 = np.array(mg20)
    # mg2 = np.array(mg2)
    # mg43 = np.array(mg43)
    # mg08 = np.array(mg08)


    #fxx functions are the results of integrating dn/dm * dm/dl * L *dl over the requisite mass range
    def f20(m): #20-100 M_sun
        return 3200.0/((-0.35) * np.log(10.))*(100.**(-0.35) - m**(-0.35))

    def f2(m): #2 - 20 M_sun
        return 0.303 * (629.9 - m**(43./20.))

    def f43(m): #0.43 - 2 M_sun
        return 0.163885 * (6.27667 - m ** (53./20.))

    def f08(m): #0.08 - 0.43 M_sun
        return 0.1051449 * (0.44853 - m ** (19./20.) )

    #all luminosity above the given mass
    a20 = f20(20.)
    a2 = f2(2.)
    a43 = f43(0.43)
    a08 = f08(0.08)

    tot = a20 + a2 + a43 + a08

    plt.plot(mg20,f20(mg20)/tot,color='b',label=r'20.0 $\leq$ $M_*$ < 100 $M_{\odot}$' + '\npiece-wise integration')
    plt.plot(mg2, ( a20 + f2(mg2))/tot, color='g',label=r'2.00 $\leq$ $M_*$ < 20.0 $M_{\odot}$'+ '\npiece-wise integration')
    plt.plot(mg43, (a20 + a2 + f43(mg43)) / tot, color='orange',label=r'0.43 $\leq$ $M_*$ < 2.00 $M_{\odot}$'+ '\npiece-wise integration')
    plt.plot(mg08, (a20 + a2 + a43 + f08(mg08)) / tot, color='r',label=r'0.08 $\leq$ $M_*$ < 0.43 $M_{\odot}$'+ '\npiece-wise integration')

    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 0.98), borderaxespad=0)

    plt.tight_layout()

    #plt.show()
    plt.savefig(op.join(OUTDIR, "hw1_p3d.png"))


def prob_3e():
    plt.close('all')
    min_mass = 0.08
    max_mass = 100.0
    mass_step = 0.01

    plt.figure()
    plt.gca().set_xscale("log")
    plt.xlim(min_mass,max_mass)
    plt.ylim(0.0,1.01)
    #plt.gca().invert_xaxis() #plot from high mass to low
    plt.title(r"Salpeter IMF ($\alpha$ = -2.35)")
    #plt.suptitle("<m> = " + str(exp_mass))
    plt.ylabel("Cumulative Luminosity Fraction $f_L$(>m)")
    plt.xlabel("$M_{*}/M_{\odot}$")


    ### 100 upper (0 aged)
    mass_grid, n_pdf, m_pdf = do_salpeter(min_mass, max_mass, mass_step)

    lum_grid = []
    for m in mass_grid:
        lum_grid.append(luminosity(m))

    lum_grid = np.array(lum_grid)

    weighted_lums = n_pdf * lum_grid  # like weighted masses

    # normalize
    weighted_lums = weighted_lums / np.sum(weighted_lums)

    # want 100% at 0.08 mass and 0% at 100.0 mass
    cdf = np.flip(np.cumsum(np.flip(weighted_lums, 0)), 0)

    plt.plot(mass_grid, cdf,color='b',label="      +0 Myr")


    ### 2.82 upper, 500 Myr aged
    #sub-select and re-normalize to what is left
    lum_grid = []
    aged_pdf = []
    aged_mass_grid = []
    for i in range(len(mass_grid)):
        if mass_grid[i] < 2.82:
            aged_mass_grid.append(mass_grid[i])
            lum_grid.append(luminosity(mass_grid[i]))
            aged_pdf.append(n_pdf[i])
        else:
            break

    lum_grid = np.array(lum_grid)
    aged_pdf = np.array(aged_pdf)
    aged_mass_grid = np.array(aged_mass_grid)

    weighted_lums = aged_pdf * lum_grid  # like weighted masses

    # normalize
    weighted_lums = weighted_lums / np.sum(weighted_lums)

    # want 100% at 0.08 mass and 0% at 100.0 mass
    cdf = np.flip(np.cumsum(np.flip(weighted_lums, 0)), 0)

    plt.plot(aged_mass_grid, cdf, color='orange',label="  +500 Myr")

    ### 2.14 upper, 1 Gyr aged
    # sub-select and re-normalize to what is left
    lum_grid = []
    aged_pdf = []
    aged_mass_grid = []
    for i in range(len(mass_grid)):
        if mass_grid[i] < 2.14:
            aged_mass_grid.append(mass_grid[i])
            lum_grid.append(luminosity(mass_grid[i]))
            aged_pdf.append(n_pdf[i])
        else:
            break

    lum_grid = np.array(lum_grid)
    aged_pdf = np.array(aged_pdf)
    aged_mass_grid = np.array(aged_mass_grid)

    weighted_lums = aged_pdf * lum_grid  # like weighted masses

    # normalize
    weighted_lums = weighted_lums / np.sum(weighted_lums)

    # want 100% at 0.08 mass and 0% at 100.0 mass
    cdf = np.flip(np.cumsum(np.flip(weighted_lums, 0)), 0)

    plt.plot(aged_mass_grid, cdf, color='r',label="+1000 Myr")

    plt.legend(loc='lower left', bbox_to_anchor=(0.1, 0.1), borderaxespad=0)

    #plt.show()
    plt.savefig(op.join(OUTDIR, "hw1_p3e.png"))


def surface_flux_lambda_to_vLv(flux,lam,radius): #assume 10pc distances
    conversion =  4 * math.pi * radius**2
    return np.float64(flux)*lam*conversion

def radius_of_star(log_g,mass):
    return math.sqrt(G_const*M_sun*mass/(10**log_g))



def read_EEM_file(mass_grid):
    # read the first 84 rows (below that, the masses are unknown, but are below our 0.08 M_sun limit anyway
    #use previously cleaned up version of the file
    t, llog, m = np.genfromtxt(op.join(BASEDIR, "EEM_dwarf_UBVIJHK_colors_Teff_DD_reduced.txt"),
                               missing_values="...",
                               max_rows=84, dtype=float, usecols=(1, 5, 18), unpack=True)
    count = 0
    for i in range(len(m)):
        if np.isnan(m[i]):
            t[i] = float('nan')
            llog[i] = float('nan')
            count += 1

    # now truncate the nan's
    m = m[count:]
    t = t[count:]
    llog = llog[count:]

    # now prepend m 100
    m = np.insert(m, 0, 100.0)
    t = np.insert(t, 0, 45000.0)
    llog = np.insert(llog, 0, 5.75)  # between the top two

    l = 10 ** llog
    t = np.flip(t, 0)
    m = np.flip(m, 0)
    l = np.flip(l, 0)

    T_grid = np.interp(mass_grid, m, t)
    L_grid = np.interp(mass_grid, m, l)

    # clean up m,t,l to original data (true for all three)
    return mass_grid, T_grid, L_grid, m, t, l

def read_fits_table(t_eff, mass):
    # return the wavlengths and corresponding fluxes (in lam*f_lam cgs units) and surface luminosity
    # based on 0 metalicity (kp00) and the T_eff determining the gravity

    filename = "kp00_"
     # maybe, show assuming <3500 == 3500 vs ignoring completely that the difference is negligible?

    # temps run 3500 to 10,000 in steps of 250
    basenum = 0
    if t_eff < 3500:
        # print("Adjusting t_eff =  %d to minimum (3500)" %t_eff)
        t_eff = 3500

    if t_eff < 10250:
        basenum = int(t_eff / 250) * 250
        mod = t_eff % 250
        if mod > 125:
            basenum += 250
        if basenum > 10000:
            basenum = 10000
    # then 10,000 to 13,000 in steps of 500
    elif t_eff < 13500:
        basenum = int(t_eff / 500) * 500
        mod = t_eff % 500
        if mod > 250:
            basenum += 500
        if basenum > 13000:
            basenum = 13000
    # then 13,000 to 35,000 in steps of 1000
    elif t_eff < 36250:
        basenum = int(t_eff / 1000) * 1000
        mod = t_eff % 1000
        if mod > 500:
            basenum += 1000
        if basenum > 35000:
            basenum = 35000
    else:
        basenum = int(t_eff / 2500) * 2500
        mod = t_eff % 2500
        if mod > 1250:
            basenum += 2500
        if basenum > 50000:
            basenum = 50000
    # then 37,500 to 50,000 in steps of 2500

    # print("Rounding T_eff = %d to %d for mass %f" %(t_eff, basenum, mass))

    filename += str(basenum) + ".fits"

    cat_loc = op.join(BASEDIR, "kp00", filename)
    table = astropy.table.Table.read(cat_loc)

    w = np.array(table['WAVELENGTH'])

    if t_eff >= 41000.:
        f = np.array(table['g50'])
        log_g = 5.0
    elif t_eff >= 36000.:
        f = np.array(table['g45'])
        log_g = 4.5
    elif t_eff >= 9000.:
        f = np.array(table['g40'])
        log_g = 4.0
    else:
        f = np.array(table['g45'])
        log_g = 4.5

    # need stars radius ....
    radius = radius_of_star(log_g, mass)
    l = surface_flux_lambda_to_vLv(f, w, radius)

    return w, f * w, l * w  # wavelengths and flux (in lam*f_lam, cgs)

def prob_4a():

    min_mass = 0.08
    max_mass = 100.0
    mass_grid, n_pdf, m_pdf = do_salpeter(min_mass, max_mass, 0.01)
    Mass, Teff, Lum, mx, tx, lx = read_EEM_file(mass_grid)  # m,t,l are the original few data

    plt.subplots(figsize=(12, 4))
    plt.subplots_adjust(wspace=0.2)
    plt.subplot(121)
    plt.title('Temperature vs Mass')
    plt.ylabel(r'$T_{eff}$ ($^\circ$K)')  # r'$\alpha > \beta$'
    plt.xlabel(r'$M_*/M_{\odot}$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylim(1e3, 1e5)

    plt.plot(Mass, Teff, color='k', linewidth=1)
    plt.scatter(mx[:-1], tx[:-1], color='b', marker='o', label="Data Points")
    plt.scatter(mx[-1], tx[-1], color='r', marker='s',label='Synthetic Data')
    plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), borderaxespad=0)


    plt.subplot(122)
    plt.title('Luminosity vs Mass')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$L_*/L_{\odot}$')
    plt.xlabel(r'$M_*/M_{\odot}$')
    plt.ylim([1e-4, 1e6])

    plt.plot(Mass, Lum, color='k', linestyle="-", alpha=1.0, linewidth=1)
    plt.scatter(mx[:-1], lx[:-1], color='b', marker='o', label="Data Points")
    plt.scatter(mx[-1], lx[-1], color='r', marker='s',label='Synthetic Data')
    plt.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), borderaxespad=0)

    plt.tight_layout()

    #plt.show()
    plt.savefig(op.join(OUTDIR, "hw1_p4a.png"))
    return



def plot_spectra_by_mass_range(mass_grid, n_pdf, Teff, spectra_grid,title,fn):

    spec = {'M': np.zeros(len(spectra_grid)),
            'FGK': np.zeros(len(spectra_grid)),
            'BA': np.zeros(len(spectra_grid)),
            'O': np.zeros(len(spectra_grid)),
            'all': np.zeros(len(spectra_grid))}

    frac = {'M': 0.0, 'FGK': 0.0, 'BA': 0.0, 'O': 0.0}

    # Mass, Teff, Lum, mass_grid, n_pdf, m_pdf all same size
    for i in range(len(mass_grid)):
        if mass_grid[i] < 0.43:  # M
            w, f, l = read_fits_table(Teff[i], mass_grid[i])
            interpolated_surface_lum = np.interp(spectra_grid, w, l)
            spec['M'] += interpolated_surface_lum * n_pdf[i]
            frac['M'] += n_pdf[i]

        elif mass_grid[i] < 2.0:  # FGK
            w, f, l = read_fits_table(Teff[i], mass_grid[i])
            interpolated_surface_lum = np.interp(spectra_grid, w, l)
            spec['FGK'] += interpolated_surface_lum * n_pdf[i]
            frac['FGK'] += n_pdf[i]

        elif mass_grid[i] < 20.0:  # BA
            w, f, l = read_fits_table(Teff[i], mass_grid[i])
            interpolated_surface_lum = np.interp(spectra_grid, w, l)
            spec['BA'] += interpolated_surface_lum * n_pdf[i]
            frac['BA'] += n_pdf[i]

        else:  # O
            w, f, l = read_fits_table(Teff[i], mass_grid[i])
            interpolated_surface_lum = np.interp(spectra_grid, w, l)
            spec['O'] += interpolated_surface_lum * n_pdf[i]
            frac['O'] += n_pdf[i]

    spec['all'] = spec['O'] + spec['BA'] + spec['FGK'] + spec['M']

    fig = plt.figure(figsize=(9,6))
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.xlim(100, 1e6)
    plt.ylim(0.01, 1e6)
    plt.title(title)
    plt.ylabel(r'$L/L_{\odot}$')
    plt.xlabel(r'$\lambda$ [$\AA$]')

    plt.plot(spectra_grid, spec['M'] / L_sun, color='red', label=r'0.08 $\leq$ $M_*$ < 0.43 $M_{\odot}$' +
                                                                 '\nfrac %0.3f' % frac['M'])  # lightest
    plt.plot(spectra_grid, spec['FGK'] / L_sun, color='orange', label=r'0.43 $\leq$ $M_*$ < 2.00 $M_{\odot}$' +
                                                                      '\nfrac %0.3f' % frac['FGK'])
    plt.plot(spectra_grid, spec['BA'] / L_sun, color='green', label=r'2.00 $\leq$ $M_*$ < 20.0 $M_{\odot}$' +
                                                                    '\nfrac %0.3f' % frac['BA'])
    plt.plot(spectra_grid, spec['O'] / L_sun, color='blue', label=r'20.0 $\leq$ $M_*$ < 100 $M_{\odot}$' +
                                                                  '\nfrac %0.3f' % frac['O'])  # heaviest
    plt.plot(spectra_grid, spec['all'] / L_sun, color='black', linestyle='dotted', label='All')

    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 0.98), borderaxespad=0)

    fig.tight_layout()
    plt.show()
    #plt.savefig(op.join(OUTDIR, fn))
    plt.close()


def prob_4b(mass_grid, n_pdf, Teff, spectra_grid):

    spec = np.zeros(len(spectra_grid))

    for i in range(len(mass_grid)):
        w, f, l = read_fits_table(Teff[i], mass_grid[i])
        interpolated_surface_lum = np.interp(spectra_grid, w, l)
        spec += interpolated_surface_lum * n_pdf[i]

    plt.figure()
    plt.gca().set_xscale("log")
    plt.gca().set_yscale("log")
    plt.xlim(100, 1e6)
    plt.ylim(0.01, 1e6)
    plt.title("Integrated Population Spectra")
    plt.ylabel(r'$L/L_{\odot}$')
    plt.xlabel(r'$\lambda$ [$\AA$]')
    plt.plot(spectra_grid, spec / L_sun, color='k')

    #plt.show()
    plt.savefig(op.join(OUTDIR, "hw1_p4b.png"))
    plt.close()


def prob_4c(mass_grid, n_pdf, Teff, spectra_grid):
    plot_spectra_by_mass_range(mass_grid, n_pdf, Teff, spectra_grid,
                               title="Integrated Population Spectra", fn="hw1_p4c.png")

def prob_4d1(mass_grid, n_pdf, Teff, spectra_grid):
    plot_spectra_by_mass_range(mass_grid, n_pdf, Teff, spectra_grid,
                               title="Integrated Population Spectra Aged 500Myr", fn="hw1_p4d1.png")

def prob_4d2(mass_grid, n_pdf, Teff, spectra_grid):
    plot_spectra_by_mass_range(mass_grid, n_pdf, Teff, spectra_grid,
                               title="Integrated Population Spectra Aged 1 Gyr", fn="hw1_p4d2.png")


def main():

    ##########################
    ##PROBLEM 2
    ##########################

    # filters = {'g':Filter('g', op.join(BASEDIR, "subaru_g.txt")),
    #            'r':Filter('r', op.join(BASEDIR, "subaru_r.txt")),
    #            'i':Filter('i', op.join(BASEDIR, "subaru_i.txt")),
    #            'z':Filter('z', op.join(BASEDIR, "subaru_z.txt")),
    #            'y':Filter('y', op.join(BASEDIR, "subaru_y.txt")),
    #            }
    #
    # star = Star('A0V', op.join(BASEDIR, "spectrum_A0V.txt"))
    #
    # prob_2a(filters, star)
    # prob_2b(filters,star)
    # #prob_2c (latex only)
    # prob_2d(star)


    ##########################
    ##PROBLEM 3
    ##########################

    #prob_3a()
    #prob_3b ...text work only
    #prob_3c ...text work only
    prob_3d()
    #prob_3e()


    ##########################
    ##PROBLEM 4
    ##########################

    # #
    #prob_4a()
    #
    # #need these for most of what follows, so just do it once
    mass_step = 1.0
    mass_grid, n_pdf, m_pdf = do_salpeter(0.08, 100.0, mass_step)
    Mass, Teff, Lum, mx, tx, lx = read_EEM_file( mass_grid)  # (min_mass,max_mass,step) #m,t,l are the original few data
    spectra_grid = np.arange(90, 1.6e6, 1)  # weighted population surface luminiosity spectra in vLv units by angstrom
    #
    # prob_4b(mass_grid=mass_grid, n_pdf=n_pdf, Teff=Teff, spectra_grid=spectra_grid)
    prob_4c(mass_grid=mass_grid, n_pdf=n_pdf, Teff=Teff, spectra_grid=spectra_grid)
    #
    # #have to rebuild for different ages (remaining masses)
    # mass_grid, n_pdf, m_pdf = do_salpeter(0.08, 2.82, mass_step)
    # Mass, Teff, Lum, mx, tx, lx = read_EEM_file(mass_grid)
    # prob_4d1(mass_grid=mass_grid, n_pdf=n_pdf, Teff=Teff, spectra_grid=spectra_grid)
    #
    # #have to rebuild for different ages (remaining masses)
    # mass_grid, n_pdf, m_pdf = do_salpeter(0.08, 2.14, mass_step)
    # Mass, Teff, Lum, mx, tx, lx = read_EEM_file(mass_grid)
    # prob_4d2(mass_grid=mass_grid, n_pdf=n_pdf, Teff=Teff, spectra_grid=spectra_grid)

    #prob_4e() ... write-up only



    exit(0)


if __name__ == '__main__':
    main()
