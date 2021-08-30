# -*- coding: utf-8 -*-
"""
author: Jack Palmer
email: jpalmer1028@gmail.com
"""

##############################################################################
################################ USER INPUTS #################################
##############################################################################

args = {
    # Title for the map
    'Plot_Title' : 'Plot of Errors',
    # Fit data export folder
    'Folder_Path' : 'C:/Users/someuser/folder_with_data',
    # Define error type. 'MSE', 'RMSE', or 'nRMSE'. 'nRMSE' is recommended
    # because it is normalized by the interquartile range allowing for comparison
    # among fits.
    'Error_Type' : 'nRMSE'
}

##############################################################################
##############################################################################
##############################################################################

"""
A script for plotting the errors of a fit. You can plot the Mean Squared Error
(MSE), Root Mean Squared Error (RMSE), or the RMSE normalized by the interquartile
range (nRMSE) of the raw data that was used in the fit. Generally, the nRMSE
is preferred because it allows for 'apples-to-apples' comparison among fits.

"""

import pandas as pd
import matplotlib.pyplot as plt  
from contextlib import suppress
import sys

def ErrorPlot(args):
    # get the data from the fit output folder
    filepath = args['Folder_Path'] + '/errors.csv'
    data = pd.read_csv(filepath)
    # Create a list to store the run numbers. This will be the x-axis
    counter = []
    for name in data.columns:
        with suppress(ValueError):
            counter.append(int(name.split(sep = '_')[-1]))
    # set the index as the error type
    data = data.set_index('error_type')
    # define the y-axis label and y_data based on desired error type
    if args['Error_Type'] == 'MSE':
        y_data = list(data.loc['MSE'])
        ylabel = 'Mean Square Error'
    elif args['Error_Type'] == 'RMSE':
        y_data = list(data.loc['RMSE'])
        ylabel = 'Root Mean Square Error'
    elif args['Error_Type'] == 'nRMSE':
        y_data = list(data.loc['nRMSE'])
        ylabel = 'Normalized Root Mean Square Error'
    # if you entered the wrong error type, the function will terminate
    else:
        sys.exit('Invalid value for Error_Type. Must be "MSE", "RMSE", or "nRMSE"')

    # define plot features
    plt.figure(dpi = 200)
    plt.title(args['Plot_Title'], fontsize = 16)
    plt.xlabel("Run Number", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    color = 'Black'
    plt.plot(counter, y_data, color = color)
    plt.tight_layout()

ErrorPlot(args)









