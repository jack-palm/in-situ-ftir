# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 09:25:42 2021

@author: jpalmer
"""

"""
A simple script to load prior fitting results. Input the folder and the script
returns the amplitudes, areas, centers, curves and raw insitu_data.

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

    return amplitudes, areas, centers, curves, insitu_data, maxima, sigmas
# call the funciton
amplitudes, areas, centers, curves, insitu_data, maxima, sigmas = LoadPriorFit('C:/Users/jpalmer/Documents/SCP/Data/FTIR/temp/FittingOutput_2021_05_10_12_08_59')


