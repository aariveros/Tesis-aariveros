from __future__ import print_function
from __future__ import division

import os, sys
import math, random
from itertools import combinations

import matplotlib.pyplot as plt
import spams

import argparse
import pickle

import numpy as np
import pandas as pd

from astropy.io import fits
from scipy.signal import savgol_filter

import packages.utils.peakdet
import packages.asydopy.db
import packages.asydopy.vu
import packages.display.db
import packages.display.vu

# ## Helper constants ###
SPEED_OF_LIGHT = 299792458.0
S_FACTOR = 2.354820045031  # sqrt(8*ln2)
KILO = 1000
DEG2ARCSEC = 3600.0

# ## ALMA Specific Constants ###
MAX_CHANNELS = 9000
MAX_BW = 2000.0  # MHz

# Without redshift (Rvel = 0)
# Temp 300 Kelvin
rvel = 0.0
temp = 300.0

ALMA_bands = {'3': [88000, 116000], '4': [125000, 163000], '6': [211000, 275000], '7': [275000, 373000],
              '8': [385000, 500000], '9': [602000, 720000]}

def gaussian(x, mu, sig):
    """

      """
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def gaussian_weighted(x, mu, sig, w):
    """

      """
    return np.power(w, 2.) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

def fwhm2sigma(freq,fwhm):
    """
      Compute the sigma in MHz given a frequency in MHz and a fwhm in km/s
      """
    sigma = (fwhm * 1000 / S_FACTOR) * (freq / SPEED_OF_LIGHT)
    return sigma

def theoretical_presence(molist, freq_init, freq_end):
    # Initialize and connect to the db
    dbpath = 'ASYDO'
    dba = packages.asydopy.db.lineDB(dbpath)
    dba.connect()
    molist_present = []
    for mol in molist:
        for iso in molist[mol]:
            linlist = dba.getSpeciesLines(iso, freq_init, freq_end)
            if linlist:
                molist_present.append(iso)
    dba.disconnect()
    return molist_present

# Function to create a fit containing an observed object (a Datacube
# ALMA-like) using ASYDO Project. Parameters:
#
#         - isolist     : list subset of the list of isotopes to generate a cube
#         - cube_name    : filename of the .fits generated by the simulation
#         - cube_params : parameters for the simulation of the cube
#         - temp
#         - rvel
#
def gen_cube(isolist, cube_params, cube_name):

    # log = sys.stdout
    dbpath = 'ASYDO'
    log = open(cube_name + '.log', 'w')
    univ = packages.asydopy.vu.Universe(log)

    for mol in isolist:

        univ.create_source('observed-'+mol, cube_params['alpha'],
                             cube_params['delta'])
        s_x = random.uniform(50, 150)
        s_y = random.uniform(40, 100)
        rot = random.uniform(10, 150)
        s_f = cube_params['s_f']
        angle = random.uniform(0,math.pi)

        model = packages.asydopy.vu.IMCM(log,dbpath, mol, temp,
                              ('normal',s_x,s_y,angle),
                              ('skew',cube_params['s_f'],cube_params['s_a']),
                              ('linear',angle,rot),
                              var_width=False)

        model.set_radial_velocity(rvel)
        univ.add_component('observed-'+mol, model)

    cube = univ.gen_cube('observerd', cube_params['alpha'],
                             cube_params['delta'], cube_params['freq'],
                             10, 20, cube_params['spe_res'],
                             cube_params['spe_bw'])

    univ.save_cube(cube, cube_name + '.fits')

def gen_cube_variable_width(isolist, cube_params, cube_name):

    # log = sys.stdout
    dbpath = 'ASYDO'
    log=open(cube_name + '.log', 'w')
    univ=packages.asydopy.vu.Universe(log)

    for mol in isolist:

        univ.create_source('observed-'+mol, cube_params['alpha'],
                             cube_params['delta'])
        s_x=random.uniform(50, 150)
        s_y=random.uniform(40, 100)
        rot=random.uniform(10, 150)
        s_f=cube_params['s_f']
        angle=random.uniform(0,math.pi)

        model=packages.asydopy.vu.IMCM(log,dbpath, mol, temp,
                              ('normal',s_x,s_y,angle),
                              ('skew',cube_params['s_f'],cube_params['s_a']),
                              ('linear',angle,rot),
                              var_width=True)

        model.set_radial_velocity(rvel)
        univ.add_component('observed-'+mol, model)

    cube = univ.gen_cube('observerd', cube_params['alpha'],
                             cube_params['delta'], cube_params['freq'],
                             10, 20, cube_params['spe_res'],
                             cube_params['spe_bw'])

    univ.save_cube(cube, cube_name + '.fits')

def gen_words(molist, cube_params, dual_words=False):
    log = sys.stdout
    dbpath = 'ASYDO'
    dictionary = pd.DataFrame([])

    last_code = ""
    last_freq = 0

    for mol in molist:
        for iso in molist[mol]:
            univ=packages.display.vu.Universe(log)
            univ.create_source('word-'+ iso)
            s_x = 1
            s_y = 1
            rot = 0
            s_f=cube_params['s_f']
            angle=math.pi
            model=packages.display.vu.IMCM(
                log,dbpath,iso,temp,
                ('normal',s_x, s_y, angle),
                ('skew', cube_params['s_f'], cube_params['s_a']),
                ('linear', angle, rot))
            model.set_radial_velocity(rvel)
            univ.add_component('word-'+ iso, model)
            lines = univ.gen_cube('observerd',
                                  cube_params['freq'],
                                  cube_params['spe_res'],
                                  cube_params['spe_bw'])
            if len(lines.hdulist) > 1:

                if(dual_words):
                    for line in lines.hdulist[1].data:
                        word =  np.array(np.zeros(len(lines.get_spectrum())))
                        '''
                            line[0] : line_code alias
                            line[1] : relative freq at the window
                        '''
                        word[line[1]] = 1
                        dictionary[line[0]] = word

                else:
                    for line in lines.hdulist[1].data:

                        last_iso = last_code.split('-')[0]
                        if(iso != last_iso and line[1] - last_freq > 2):
                            word =  np.array(np.zeros(len(lines.get_spectrum())))
                            '''
                                line[0] : line_code alias
                                line[1] : relative freq at the window
                            '''
                            word[line[1]] = 1
                            dictionary[line[0]] = word
                        else:
                            dictionary.pop(last_code)
                            word[line[1]] = 1

                            dual_alias = last_code + "&&" + \
                                         line[0].split('-')[1]

                            dictionary[dual_alias] = word
                        last_code = line[0]
                        last_freq = line[1]


    dictionary.index = get_freq_index_from_params(cube_params)
    return dictionary

def get_freq_index_from_params(cube_params):
    return np.arange(cube_params['freq'] - int(cube_params['spe_bw']/2),
                     cube_params['freq'] + int(cube_params['spe_bw']/2),
                     cube_params['spe_res'])

def save_dictionary(D, band):
    output = open('pickle/dictionary_' + band + '.pkl', 'wb')
    pickle.dump(D, output)
    output.close()

def load_dictionary(band):
    input_file = open('pickle/dictionary_' + band + '.pkl', "rb" )
    D = pickle.load( input_file )
    input_file.close()
    return D

def save_isolist(D):
    output = open('pickle/isolist.pkl', 'wb')
    pickle.dump(D, output)
    output.close()


def load_isolist():
    input_file = open('pickle/isolist.pkl', "rb" )
    D = pickle.load(input_file)
    input_file.close()
    return D

def get_fortran_array(input):
    fort_array = np.asfortranarray(np.asmatrix(input)).T
    fort_array = np.asfortranarray(fort_array, dtype= np.double)
    return fort_array

def show_alphas(alpha):
    for p in xrange(0, len(alpha)):
        if alpha[p] != 0:
            print(dictionary.columns[p] + ": " +  str(alpha[p]))

def show_words(dictionary):
    for iso in dictionary.columns:
        plt.plot(dictionary[iso])
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

def graph_sparse_coding(Detector, dictionary_recal, alpha,
                        cube_params, y_train, total):
    lines = Detector.get_lines_from_fits()

    # Catches the predicted lines
    alpha = pd.Series(alpha[:,0])
    alpha.index = dictionary_recal.columns
    alpha = alpha[alpha > 0]


    for line in lines:
        # Shows lines really present
        isotope_frequency = int(line[1])
        isotope_name = line[0] + "-f" + str(line[1])

        plt.axvline(x=isotope_frequency, ymin=0, ymax= 3, color='g')
        plt.text(isotope_frequency, 22.0, isotope_name, size='14', rotation='vertical')

        # Show predicted classes
        tot_sum = 0
        aux_alpha = pd.Series([])
        for line_name in alpha.index:
            if dictionary_recal[line_name].loc[isotope_frequency] != 0:
                aux_alpha[line_name] = alpha[line_name]
                tot_sum += np.absolute(alpha[line_name])
        aux_alpha = aux_alpha/tot_sum

        i = 2
        for line_name in aux_alpha.index:
            plt.axvline(x=isotope_frequency, ymin=0, ymax= 0, color='g')
            plt.text(isotope_frequency, 0.15 + i, line_name, size='14', rotation='vertical')
            plt.text(isotope_frequency + 33, 0.15 + i, str(aux_alpha[line_name]), size='14', rotation='vertical')
            i += 4

    xaxis = Detector.get_freq_index_from_params()
    plt.plot(xaxis, y_train*20, color='r', label='Observed')
    plt.plot(xaxis, total*20, color='b', label='Recovered', linestyle='--')
    plt.legend(loc='upper right')
    plt.xlim(xmax = xaxis[-1], xmin = xaxis[0])
    plt.ylim(ymax = 25, ymin = -1)
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


def fill_precision(Results, MatrixConfusion,):
    for isotope in MatrixConfusion.columns:
        true_positives = 0.0
        tot = 0.0
        for row in MatrixConfusion.index:
            if isotope == row:
                true_positives += MatrixConfusion.loc[row][isotope]
            tot += MatrixConfusion.loc[row][isotope]
        if tot != 0:
            Results['Precision'].loc[isotope] = 1.0*true_positives/tot
    return Results

def fill_recall(Results, MatrixConfusion):
    for isotope in MatrixConfusion.columns:
        if isotope not in MatrixConfusion.index:
            Results['Recall'].loc[isotope] = 0
            continue

        true_positives = 0.0
        tot = 0.0
        for column in MatrixConfusion.columns:
            if isotope == column:
                true_positives += MatrixConfusion.loc[isotope][column]
            tot += MatrixConfusion.loc[isotope][column]
        if tot != 0:
            Results['Recall'].loc[isotope] = 1.0*true_positives/tot
    return Results

def fill_fscore(Results, MatrixConfusion):
    for isotope in MatrixConfusion.columns:
        recall = Results['Recall'].loc[isotope]
        precision = Results['Precision'].loc[isotope]
        if recall != 0 or precision != 0:
            Results['F-Score'].loc[isotope] = 2.*(recall*precision)/(recall + precision)
    return Results

def get_results(confusion_matrix):
    results = pd.DataFrame(np.zeros((len(confusion_matrix.columns), 3)),
                      index=confusion_matrix.columns,
                      columns=['Precision', 'Recall', 'F-Score'])*1.0
    results = fill_precision(results, confusion_matrix)
    results = fill_recall(results, confusion_matrix)
    results = fill_fscore(results, confusion_matrix)
    return results


def get_confusion_matrix(dictionary_recal, alpha, file_path, cube_params):

    lines = get_lines_from_fits(file_path)

    # Catches the predicted lines
    alpha_columns = pd.Series(alpha[:,0])
    alpha_columns.index = dictionary_recal.columns
    alpha_columns = alpha_columns[alpha_columns > 0]

    set_isotopes = set()
    # Catches the lines really present
    for idx in range(0, len(lines)):
        set_isotopes.add(lines.values[idx][0] + "-f" + str(lines.values[idx][1]))

    # Confusion Matrix construction
    confusion_matrix = pd.DataFrame(np.zeros((len(alpha_columns),
                                            len(set_isotopes))),
                                    index=alpha_columns.index, columns=set_isotopes)

    for idx in  range(0, len(lines)):
        isotope_name = lines.values[idx][0] + "-f" + str(lines.values[idx][1])

        tot_sum = 0
        aux_alpha = pd.Series([])
        for line_name in alpha_columns.index:
            if dictionary_recal[line_name].loc[lines.index[idx]] != 0:
                aux_alpha[line_name] = alpha_columns[line_name]
                tot_sum += np.absolute(alpha_columns[line_name])
        aux_alpha = aux_alpha/tot_sum

        for isotope in aux_alpha.index:
            confusion_matrix[isotope_name].loc[isotope] += aux_alpha[isotope]

    return confusion_matrix


def test(dictionary_recal, alpha, file_path, cube_params):

    confusion_matrix = get_confusion_matrix(dictionary_recal, alpha,
                                           file_path, cube_params)

    results = get_results(confusion_matrix)

    return confusion_matrix, results

def print_predictions(Detector, confusion_matrix):
    lines = Detector.get_lines_from_fits()

    for line in lines:
        isotope_frequency = int(line[1])
        isotope_name = line[0] + "-f" + str(line[1])
        isotope_temp = line[2]

        print("# " + isotope_name + ", temperature: " + str(isotope_temp))
        for predicted_isotope in confusion_matrix.index:
            if confusion_matrix.loc[predicted_isotope][isotope_name] != 0:
                print(predicted_isotope + ": " + str(confusion_matrix.loc[predicted_isotope][isotope_name]))

#####
#
####

def get_thresold_parameter(file_path, cube_params):
    mean_noise, std_noise = get_noise_parameters_from_fits(file_path, cube_params)
    return max(mean_noise, 0) + 3*std_noise

def detect_lines(file_path, cube_params, option=''):

    detected_lines = np.zeros([cube_params['spe_bw']/cube_params['spe_bw']])
    detected_temps = np.zeros([cube_params['spe_bw']/cube_params['spe_bw']])

    values = get_values_filtered_normalized(file_path, (1,1), cube_params)

    mean_noise = get_noise_parameters_from_fits(file_path, cube_params)
    threshold = get_thresold_parameter(file_path, cube_params)

    # Array with max and min detected in the spectra
    maxtab, mintab = packages.utils.peakdet.peakdet(values, threshold)

    for max_line_temp in maxtab[:,1]:

        if max_line_temp > np.max(mean_noise, 0):

            max_line_freq = maxtab[maxtab[:,1] == max_line_temp][:,0]

            # Set 1 as value of the line
            detected_lines[int(max_line_freq)] = 1

            # Save the temp for the property
            detected_temps[int(max_line_freq)] += max_line_temp

    if option == 'temp':    return detected_temps

    return detected_lines

def detect_lines_subtracting_gaussians(file_path, cube_params, option=''):

    detected_lines = np.zeros([cube_params['spe_bw']/cube_params['spe_res']])
    detected_temps = np.zeros([cube_params['spe_bw']/cube_params['spe_res']])

    values = get_values_filtered_normalized(file_path, (1,1), cube_params)

    mean_noise, std_noise = get_noise_parameters_from_fits(file_path, cube_params)
    threshold = get_thresold_parameter(file_path, cube_params)

    # Array with max and min detected in the spectra
    maxtab, mintab = packages.utils.peakdet.peakdet(values, threshold)
    gauss_domain = np.arange(0, cube_params['spe_bw']/cube_params['spe_res'], 1)

    while len(maxtab) > 0:
        max_line_temp = np.max(maxtab[:,1])

        if max_line_temp > max(mean_noise, 0):

            max_line_freq = maxtab[maxtab[:,1] == max_line_temp][:,0]

            # Fit the gaussian
            gauss_fitt = (max_line_temp)*gaussian(gauss_domain,
                    max_line_freq, cube_params['s_f'])

            # Subtract the gaussian
            for i in range(0, len(values)):
                values[i] = max(values[i] - gauss_fitt[i], mean_noise)

            # Set 1 as value of the line
            detected_lines[int(max_line_freq)] = 1
            # Save the temp for the property
            detected_temps[int(max_line_freq)] += max_line_temp

            maxtab, mintab = packages.utils.peakdet.peakdet(values,  threshold)
        else:
            break

    if option == 'temp':    return detected_temps
    return detected_lines

def get_lines_from_fits(file_path):
    lines = pd.Series([])

    i = 3
    hdu_list = fits.open(file_path)
    while(i < len(hdu_list)):
        for line in hdu_list[i].data:
            """
                line[1] : Formula
                line[3] : Frequency (MHz)
                line[6] : Temperature (No unit)
            """
            lines[int(line[3])]= [line[1], line[3]]
        i = i + 3
    hdu_list.close()
    return lines

def get_temps_from_fits(file_path):
    lines = pd.DataFrame([])
    i = 3
    hdu_list = fits.open(file_path)
    while(i < len(hdu_list)):
        for line in hdu_list[i].data:
            """
                line[1] : Formula
                line[3] : Frequency (MHz)
                line[6] : Temperature (No unit)
            """
            lines.loc[int(line[3])]= [line[1], line[6]]
        i = i + 3
    hdu_list.close()
    return lines

def near_obs_freq(freq_theo, file_path, cube_params, detected_lines):

    min_dist = cube_params['spe_bw']/cube_params['spe_res']
    near_freq = 0

    for freq_obs in range(0, int(min_dist)):
        if detected_lines[freq_obs] != 0:
            dist = math.fabs(freq_theo - freq_obs)
            if dist == 0:
                return freq_obs
            elif min_dist > dist:
                min_dist = dist
                near_freq = freq_obs
    return near_freq


def near_obs_prob(freq_theo, near_freq_obs, file_path, cube_params,
                  detected_temps):
    sigma = fwhm2sigma(cube_params['freq'], cube_params['s_f'])

    gauss_weight = gaussian_weighted(freq_theo, near_freq_obs,
                                  sigma, detected_temps[near_freq_obs])
    factor = 2*sigma
    ini = int(round(freq_theo - factor))
    end =int(round(freq_theo + factor))
    if ini < 0:
        ini = 0
    size_spectra = cube_params['spe_bw']/cube_params['spe_res']
    if end > size_spectra:
        end = size_spectra
    window = np.arange(ini, end, 1)

    gauss_fit = gauss_weight*gaussian(window, near_freq_obs, sigma)
    return gauss_fit, window

def recal_words(file_path, words, cube_params):

    words_recal = pd.DataFrame(np.zeros(words.shape))
    words_recal.index = np.arange(0,
                                  cube_params['spe_bw']/cube_params['spe_res'],
                                  1)
    words_recal.columns = words.columns

    size_spectra = cube_params['spe_bw']/cube_params['spe_res']

    detected_lines = detect_lines_subtracting_gaussians(file_path, cube_params)

    detected_temps = detect_lines_subtracting_gaussians(file_path, cube_params,
                                                        option="temp")

    for mol in words.columns: #'CH3OHvt=0-f602233.197'
        # The theorethical line will be replaced by the max probability of
        # the nearest observed line (Gaussian decay distance weighted)
        for freq_theo in range(0, int(size_spectra)):
            if words.iloc[freq_theo][mol] != 0: #117
                nof = near_obs_freq(freq_theo, file_path, cube_params,
                                    detected_lines)
                gauss_fit, window = near_obs_prob(freq_theo, nof,
                                                  file_path, cube_params,
                                                  detected_temps)
                # Reeplace the highest probability for each theoretical line
                words_recal[mol].iloc[window] = gauss_fit
                break
    words_recal.index = words.index
    return words_recal, detected_lines

def get_data_from_fits(file_path):
    hdu_list = fits.open(file_path)
    data = np.array(hdu_list[0].data)
    hdu_list.close()
    return data

def get_win_len_from_s_f(cube_params):
    # Calculating some other params
    sigma = fwhm2sigma(cube_params['freq'], cube_params['s_f'])
    win_len = int(sigma)
    # The width of the windows must be odd
    if (win_len % 2) == 0:
        win_len -= 1
    return win_len

def get_values_filtered_normalized(file_path, train_pixel, cube_params):
    data = get_data_from_fits(file_path)

    # Pre-processing
    # The training pixel values
    values = data[:, train_pixel[0], train_pixel[1]]

    # Apllying a Savitzky-Golay first derivative
    # n#-sigma-point averaging algorithm is applied to the raw dataset
    win_len = get_win_len_from_s_f(cube_params)

    # Parameter to reduce noise
    poli_order = 2
    values = savgol_filter(values, win_len, poli_order)
    # Normalize by the maximum of the serie
    values = values/np.max(values)
    return values


def get_noise_parameters_from_fits(file_path, cube_params):
    data = get_data_from_fits(file_path)
    win_len = get_win_len_from_s_f(cube_params)

    # The pixel (0, 0) always will be a empty pixel with noise
    values_noise = data[:,0,0]
    values = data[:,0,1]
    poli_order = 2
    values = savgol_filter(values, win_len, poli_order)

    values_noise = savgol_filter(values_noise, win_len, poli_order)
    values_noise = values_noise/np.max(values)
    mean_noise = np.mean(values_noise)
    std_noise = np.std(values_noise)
    return mean_noise, std_noise

def plot_data(data, point):
    values = data[:,point[0],point[1]]
    plt.plot(values, color='r', label='Observed spectra')
    plt.xlabel('Relative Frequency [MHz]')
    plt.ylabel('Temperature [Normalized]')
    plt.legend(loc='upper right')
    plt.show()

def plot_detected_lines(self):
    # Plot detected max
    plt.plot(self.values, color='r', label='Observed Filtered')
    plt.xlabel('Relative Frequency [MHz]')
    plt.ylabel('Temperature [Normalized]')
    plt.legend(loc='upper right')
    # Plot Array with max and min detected in the spectra
    # Plot max point
    for i in range(len(self.max_line_freq)):
        plt.plot(self.max_line_freq[i], self.max_line_temp[i], 'bs')
    plt.axhspan(0, self.threshold, alpha=0.5)
    plt.show()

molist = {
            'CO' : ('COv=0','COv=1','13COv=0','C18O','C17O','13C17O','13C18O'),
            # Carbon Monoxide

            # 'NH2' : ('NH2'), # Amidogen

            'N2H' : ('N2H+v=0', 'N2D+', '15NNH+', 'N15NH+'), # Diazenylium

            'CN' : ('CNv=0', '13CN', 'C15N'), # Cyanide Radical

            'HCN' : ('HCNv=0', 'HCNv2=1', 'HCNv2=2','HCNv3=1', 'HC15Nv=0',
                     'H13CNv2=1', 'H13CNv=0', 'HCNv1=1', 'DCNv=0',
                     'DCNv2=1', 'HCNv2=4', 'HCNv2=1^1-v2=4^0'),
            # Hydrogen Cyanide

            # 'H2CN' : ('H2CN'), # Methylene amidogen

            'CS' : ('CSv=0', '13C34Sv=0', 'C36Sv=0', 'C34Sv=0', 'CSv=1-0',
                    '13CSv=0', 'C33Sv=0', 'CSv=1', 'C34Sv=1'),
            # Carbon Monosulfide

            'CCS' : ('CCS', 'C13CS', '13CCS', 'CC34S'), # Thioxoethenylidene

            'H2S' : ('H2S', 'H234S', 'D2S'), # Hydrogen sulfide

            'H2CS' : ('H2CS', 'H213CS', 'H2C34S'), # Thioformaldehyde

            'SO2' : ('SO2v=0', '33SO2', '34SO2v=0', 'SO2v2=1', 'OS18O',
                    'OS17O'),
            # Sulfur Dioxide

            'H2CO' : ('H2CO', 'H2C18O', 'H213CO'), # Formaldehyde

            'HCO' : ('HCO+v=0', 'HC18O+', 'HC17O+', 'H13CO+'), # Formylium

            # 'HC3N' : ('HC3Nv=0'), # Cyanoacetylene

            'HC5N' : ('HC5Nv=0', 'HC5Nv11=1', 'HCC13CCCN', 'HCCCC13CN',
                      'HCCC13CCN', 'H13CCCCCN'), # Cyanobutadiyne

            'CH3OH' : ('CH3OHvt=0', '13CH3OHvt=0 ', 'CH318OH', 'CH3OHvt=1 ',
                       '13CH3OHvt=1 ') # Methanol
          }

if __name__ == '__main__':
    pass
