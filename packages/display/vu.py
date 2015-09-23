# import random
import numpy as np
import pylab as plt
import matplotlib.animation as animation
from astropy.io import fits
from scipy import pi, sqrt, exp
import math
import db
from scipy.special import erf



# ## Helper constants ###
SPEED_OF_LIGHT = 299792458.0
S_FACTOR = 2.354820045031  # sqrt(8*ln2)
KILO = 1000
DEG2ARCSEC = 3600.0

# ## ALMA Specific Constants ###
MAX_CHANNELS = 9000
MAX_BW = 2000.0  # MHz

### Toolbox functions ###

def freq_window(freq, axis):
    """
    Compute a window centered at freq within an axis.

    freq_window() returns a tuple with the lower and upper indices in the axis for a
    frequency window of freq +- factor. The size is at most 2*factor, but
    is limited by the axis borders. It returns (0,0) or (end,end) if the window is
    out of the axis by the left or right respectively (i.e., end = len(axis)).
    Equispaced axes are assumed.
    @rtype : tuple
    @param freq: center frequency
    @param factor: the window factor (half window size)
    @param axis: a equispaced array of elements
    """
    dlta = axis[1] - axis[0]
    index = int(round((freq - axis[0]) / dlta))
    if index < 0:
        index = 0
    if index > len(axis):
        index = len(axis)
    return index

### Core ASYDO Classes ###

class Universe:
    """
    A synthetic universe where to put synthetic objects.
    """

    def __init__(self, log):
        """
        The log parameter is an opened file descriptor for logging purposes
        """
        self.log = log
        self.sources = dict()

    def create_source(self, name):
        """
        A source needs a name.
        """
        self.sources[name] = Source(self.log, name)

    def add_component(self, source_name, model):
        """
        To add a component a Component object must be instantiated (model), and added to
        a source called source_name.
        """
        self.sources[source_name].add_component(model)

    def gen_cube(self, name, freq, spe_res, spe_bw):
        """
        Returns a SpectralCube object where all the sources within the FOV and BW are projected.

        This function needs the following parameters:
        - name    : name of the cube
        - freq    : spectral center (frequency)
        - spe_res : spectral resolution
        - spe_bw  : spectral bandwidth
        """
        cube = SpectralCube(self.log, name, freq, spe_res, spe_bw)
        for src in self.sources:
            self.sources[src].project(cube)
        return cube

    def save_cube(self, cube, filename):
        """
        Wrapper function that saves a cube into a FITS (filename).
        """
        # self.log.write('   -++ Saving FITS: ' + filename + '\n')
        cube.save_fits(self.sources, filename)

    def remove_source(self, name):
        """
        Deletes a source and its components.
        """
        # self.log.write('Removing source ' + name)
        return self.sources.remove(name)


class Source:
    """
    A generic source of electromagnetic waves with several components.
    """

    def __init__(self, log, name):
        """ Parameters:
               * log: logging descriptor
               * name: a name of the source
        """
        self.log = log
        self.name = name
        self.comp = list()

    def add_component(self, model):
        """ Defines a new component from a model.
        """
        code = self.name + '-c' + str(len(self.comp) + 1)  #+ '-r' + str(self.alpha) +'-d'+str(self.delta)
        self.comp.append(model)
        model.register(code)


    def project(self, cube):
        """
        Projects all components in the source to a cube.
        """
        for component in self.comp:
            # self.log.write('  |- Projecting ' + component.comp_name + '\n')
            component.project(cube);


class SpectralCube:
    """
    A synthetic spectral cube.
    """

    def __init__(self, log, name, freq, spe_res, spe_bw):
        """
        Obligatory Parameters:
        - log	  : descriptor of a log file
        - name    : name of the cube
        - freq    : spectral center (frequency)
        - spe_res : spectral resolution
        - spe_bw  : spectral bandwidth

        Optional Parameters:
        - band_freq   : a diccionary of frequency ranges for the bands (key = band_name, value = (lower,upper))
        """
        self.name = name
        self.freq = freq
        self.spe_res = spe_res
        self.spe_bw = spe_bw

        self.freq_border = [freq - spe_bw / 2.0, freq + spe_bw / 2.0]
        self.channels = round(spe_bw / spe_res)
        self.freq_axis = np.linspace(self.freq_border[0], self.freq_border[1], self.channels)
        self.data = (
                        np.zeros(
                            (len(self.freq_axis))) )
        self.hdulist = fits.HDUList([self._get_cube_HDU()])

    def get_spectrum(self):
        """ Returns the spectrum of a (x,y) position """
        return self.data


    def _get_cube_HDU(self):
        prihdr = fits.Header()
        prihdr['AUTHOR'] = 'Astronomical SYnthetic Data Observatory'
        prihdr['COMMENT'] = "Here's some commentary about this FITS file."
        prihdr['SIMPLE'] = True
        # prihdr['BITPIX'] = 8
        prihdr['NAXIS'] = 3
        hdu = fits.PrimaryHDU(header=prihdr)
        hdu.data = self.data
        return hdu

    def _add_HDU(self, hdu):
        pass
        self.hdulist.append(hdu)

    def save_fits(self, sources, filename):
        """ Simple as that... saves the whole cube """
        self.hdulist.writeto(filename, clobber=True)

    def _updatefig(self, j):
        """ Animate helper function """
        self.im.set_array(self.data[j, :, :])
        return self.im,

    def animate(self, inte, rep=True):
        """ Simple animation of the cube.
            - inte       : time interval between frames
            - rep[=True] : boolean to repeat the animation
          """
        fig = plt.figure()
        self.im = plt.imshow(self.data[0, :, :], cmap=plt.get_cmap('jet'), vmin=self.data.min(), vmax=self.data.max(), \
                             extent=(
                                 self.alpha_border[0], self.alpha_border[1], self.delta_border[0],
                                 self.delta_border[1]))
        ani = animation.FuncAnimation(fig, self._updatefig, frames=range(len(self.freq_axis)), interval=inte, blit=True,
                                      repeat=rep)
        plt.show()


class Component:
    """Abstract component model"""

    def __init__(self, log, z_base=0.0):
        """ log: file descriptor for logging
            z_base[=0] : optional parameter to set base redshift (if not, please use set_radial_vel)
        """
        self.log = log
        self.z = z_base
        self.rv = SPEED_OF_LIGHT / KILO * ((self.z ** 2 + 2 * self.z) / (self.z ** 2 + 2 * self.z + 2))

    def set_radial_velocity(self, rvel):
        """Set radial velocity rvel in km/s"""
        self.rv = rvel
        self.z = math.sqrt((1 + self.rv * KILO / SPEED_OF_LIGHT) / (1 - self.rv * KILO / SPEED_OF_LIGHT)) - 1

    def register(self, comp_name):
        """Register the component filename """
        self.comp_name = comp_name

    def project(self, cube):
        """Project the component in the cube"""
        pass


class IMCM(Component):
    """ Interstellar Molecular Cloud Model """

    def __init__(self, log, dbpath, mol_list, temp, spa_form, spe_form, z_grad, z_base=0.0, abun_max=10 ** -5,
                 abun_min=10 ** -6, abun_CO=1.0):
        Component.__init__(self, log, z_base)
        self.spa_form = spa_form
        self.spe_form = spe_form
        self.z_grad = z_grad
        self.dbpath = dbpath
        self.temp = temp
        self.intens = dict()
        for mol in mol_list.split(','):
            self.intens[mol] = 1



    def project(self, cube):
        arr_code = []
        arr_rest_freq = []
        dba = db.lineDB(self.dbpath)
        dba.connect()
        freq_init_corr = cube.freq_border[0] / (1 + self.z)
        freq_end_corr = cube.freq_border[1] / (1 + self.z)
        count = 0
        used = False
        for mol in self.intens:
            # For each molecule specified in the dictionary
            # load its spectral lines

            linlist = dba.getSpeciesLines(mol, freq_init_corr,
                                          freq_end_corr)  # Selected spectral lines for this molecule

            for lin in linlist:
                count += 1
                freq = (1 + self.z) * lin[3]  # Catalogs must be in Mhz
                window = freq_window(freq, cube.freq_axis)
                cube.data[window] = 1
                used = True

                arr_code.append(mol + "-f" + str(lin[3]))
                arr_rest_freq.append(window)
        dba.disconnect()
        if not used:
            return
        tbhdu = fits.new_table(fits.ColDefs([
        fits.Column(name='line_code', format='40A', array=arr_code), \
        fits.Column(name='line', format='D', array=arr_rest_freq) \
        ]))
        cube._add_HDU(tbhdu)
