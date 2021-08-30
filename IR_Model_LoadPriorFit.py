# -*- coding: utf-8 -*-
"""
author: Jack Palmer
email: jpalmer1028@gmail.com
"""

##############################################################################
################################ USER INPUTS #################################
##############################################################################

args= {# String of absolute path
       'folder_path' : 'C:/Users/someuser/folder_with_fit_data'
}

##############################################################################
##############################################################################
##############################################################################

"""
A simple script to load prior fitting results. Input the folder and the script
returns the amplitudes, areas, centers, curves, maxima, sigmas, errors and raw insitu_data.

"""

def LoadPriorFit(folder):
    
    import pandas as pd
    import numpy as np
    # read in the .csv files
    amplitudes = pd.read_csv(folder + '/amplitudes.csv')
    areas = pd.read_csv(folder + '/areas.csv')
    centers = pd.read_csv(folder + '/centers.csv')
    sigmas = pd.read_csv(folder + '/sigmas.csv')
    maxima = pd.read_csv(folder + '/maxima.csv')
    insitu_data = pd.read_csv(folder + '/insitu_dataset.csv')
    # read in the curves dictionary
    curves = np.load(folder + '/curves_dict.npy',allow_pickle='TRUE').item()
    errors = np.load(folder + '/errors_dict.npy',allow_pickle='TRUE').item()

    return amplitudes, areas, centers, curves, insitu_data, maxima, sigmas, errors

# call the funciton
amplitudes, areas, centers, curves, insitu_data, maxima, sigmas, errors = LoadPriorFit(args['folder_path'])


