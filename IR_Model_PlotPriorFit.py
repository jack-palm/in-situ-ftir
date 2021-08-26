# -*- coding: utf-8 -*-
"""
author: Jack Palmer
email: jpalmer1028@gmail.com
"""

##############################################################################
################################ USER INPUTS #################################
##############################################################################

args = {# Fitting results output folder
        'folder' : 'C:/Users/someuser/folder_with_data',
        # Any "run number" within the fit output. Integer.
        'start' : 1,
        # An integer or 'all'. All will plot all spectra from the fit
        'amount' : 'all'
}

##############################################################################
##############################################################################
##############################################################################

def PlotPriorFit(args):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pl
    import os
    from natsort import natsorted
    
    # store filenames
    files = os.listdir(args['folder']+'/curves')
    
    # Create a list keys and store the file names minus '.csv' there
    keys = []
    for i in range(len(files)):
        keys.append(files[i][:-4])
 
    #sort the list keys in a natural order
    keys = natsorted(keys)
    
    # Create a list 'counter' and populate with the spectrum number
    counter = []
    for i in range(len(files)):
        counter.append(keys[i].split(sep='_')[-1])
        
    # Extract the desired data from the curves folder and store to the dict 'curves'
    curves = {}
    if args['amount'] == 'all' or args['amount'] == 'All':
        for i in range(len(files)):
            curves[keys[i]] = pd.read_csv(args['folder'] + '/curves/' + files[i])
    else:
        # Define indices to retrieve desired data
        ind1 = counter.index(str(args['start']))
        ind2 = ind1 + args['amount']
        for i in np.arange(ind1,ind2):
            curves[keys[i]] = pd.read_csv(args['folder'] + '/curves/' + files[i])
    # iterate over curves dictionary to plot everything
    for key in curves.keys():
        
        plt.figure(figsize=(4.5,4)) 
        plt.figure(dpi = 200)
        plt.xlabel("Wavenumber ($cm^{-1}$)", fontsize=12)
        plt.ylabel("Absorbance (a.u.)", fontsize=12)
        # create a color scheme
        colors = pl.cm.jet(np.linspace(0,1,len(curves[key].columns)-3))
        cols = list(curves[key].columns)
        cols.sort()
        # iteratively add all components to the plot      
        for i in np.arange(1,len(cols)-2):
            plt.plot(curves[key]['Wavenumber'], 
                     curves[key][cols[i]], 
                     label = cols[i],
                     color=colors[i-1])
            # shade the area under the curve
            plt.fill_between(curves[key]['Wavenumber'], 
                             0,
                             curves[key][cols[i]],
                             alpha=0.3,
                             color=colors[i-1])        
        # add the raw data to the plot
        plt.plot(curves[key]['Wavenumber'], curves[key]['Raw_Data'], linewidth=2, label='Raw Data',color = 'hotpink')
        # add the best fit to the plot
        plt.plot(curves[key]['Wavenumber'],curves[key]['Best_Fit'], '--', label='Best Fit',alpha = 0.5,color = 'black')
        plt.title(key)
        plt.xlim(max(curves[key]['Wavenumber']),min(curves[key]['Wavenumber']))
        plt.legend(fontsize=5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

PlotPriorFit(args)







    
    
    
    
    
    
    
    
    
    
    