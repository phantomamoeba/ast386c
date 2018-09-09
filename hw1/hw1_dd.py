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


BASEDIR = "res"
OUTDIR = "out"

c_const = 2.99792458e18 #angstroms/sec
pc_cm = 3.086e18
L_sun = 3.839e33

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
        # w_i = min(filter.w)
        # w_f = max(filter.w)
        # top = quad(top_int,w_i,w_f,args=(f_w,filter.Tw))

        # top = 0.
        # bot = 0.
        # mid = np.median(filter.w)
        #
        # for i in range(len(self.w)-1):
        #     top += self.w[i] * self.Sw[i]*filter.getTw(self.w[i])*(self.w[i+1]-self.w[i]) #* (self.w[i]**2) / c_const
        #     bot += self.w[i] * filter.getTw(self.w[i]) * (self.w[i + 1] - self.w[i]) #* (self.w[i]**2) / c_const
        #
        #
        # fflux_v = top/bot /( c_const/(mid**2)) #should be iso not mid, but this is close
        #convert to fflux_v
        #

        #better ... since this is discrete, though, just run two sums rather than two integrals
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

    fig = plt.figure(figsize=(8,5))
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
    plt.xlabel('Wavelength ($\AA$)')

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


def main():

    filters = {'g':Filter('g', op.join(BASEDIR, "subaru_g.txt")),
               'r':Filter('r', op.join(BASEDIR, "subaru_r.txt")),
               'i':Filter('i', op.join(BASEDIR, "subaru_i.txt")),
               'z':Filter('z', op.join(BASEDIR, "subaru_z.txt")),
               'y':Filter('y', op.join(BASEDIR, "subaru_y.txt")),
               }

    star = Star('A0V', op.join(BASEDIR, "spectrum_A0V.txt"))

    prob_2a(filters, star)
    prob_2b(filters,star)
    #prob_2c (latex only)
    prob_2d(star)


if __name__ == '__main__':
    main()
