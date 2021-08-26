# -*- coding: utf-8 -*-
"""
author: Jack Palmer
email: jpalmer1028@gmail.com
"""

##############################################################################
################################ USER INPUTS #################################
##############################################################################

args = { 
   # Enter the paths for the fit results and the corresponding EChem data    
    'EChem_file' : 'C:/Users/someuser/folder_with_echem_data/echem_data.txt',
    'Fit_folder' : 'C:/Users/someuser/folder_with_fit_data',  
    # Enter a single integer corresponding to the component you would like to plot
    'component_to_plot' : 591 # 503, 515, 544, 623, 567, 580, 591
}

##############################################################################
##############################################################################
##############################################################################


def AreaCenterMax_vs_Voltage():
    
    import pandas as pd
    import matplotlib.pyplot as plt  
    from scipy import interpolate
    # from tkinter import filedialog as fd
    # EChem_file = fd.askopenfilename()           
    # Fit_folder = fd.askdirectory()
    
    # Import the echem data as and make the time column numeric
    EChem_data = pd.read_csv(args['EChem_file'],skiprows = 2,sep='\t')   
    EChem_data['TestTime'] = EChem_data['TestTime'].str.replace(",","").astype(float)
    # Import the fit results
    areas = pd.read_csv(args['Fit_folder'] + '/areas.csv')
    centers = pd.read_csv(args['Fit_folder'] + '/centers.csv')
    maxima = pd.read_csv(args['Fit_folder'] + '/maxima.csv')
    # extract the spectrum number from the title and store to Spec_Time
    Spec_Time = []
    for i in range(len(areas.columns)-1):
        Spec_Time.append(int(areas.columns[i+1].split(sep='_')[1]))
    # transpose the fit data and reset x and y indices
    areas = areas.transpose()
    areas = areas.reset_index()
    areas = areas.drop(columns = 'index')
    areas.columns = areas.iloc[0]
    areas = areas.drop(areas.index[0])
    areas = areas.reset_index()
    areas = areas.drop(columns = 'index')
    centers = centers.transpose()
    centers = centers.reset_index()
    centers = centers.drop(columns = 'index')
    centers.columns = centers.iloc[0]
    centers = centers.drop(centers.index[0])
    centers = centers.reset_index()
    centers = centers.drop(columns = 'index')
    maxima = maxima.transpose()
    maxima = maxima.reset_index()
    maxima = maxima.drop(columns = 'index')
    maxima.columns = maxima.iloc[0]
    maxima = maxima.drop(maxima.index[0])
    maxima = maxima.reset_index()
    maxima = maxima.drop(columns = 'index')
    # add the column Spec_Time to the df's
    areas['Spec_Time'] = Spec_Time
    centers['Spec_Time'] = Spec_Time
    maxima['Spec_Time'] = Spec_Time
    # Set all time columns in terms of hours. It is assumed that an FTIR spectrum
    # is started at t = 0. EChem data is assumed to be reported in seconds.
    areas['Spec_Time'] = ((areas['Spec_Time']*args['spectrum_interval']) - args['spectrum_interval'])/60
    centers['Spec_Time'] = ((centers['Spec_Time']*args['spectrum_interval']) - args['spectrum_interval'])/60
    maxima['Spec_Time'] = ((maxima['Spec_Time']*args['spectrum_interval']) - args['spectrum_interval'])/60
    EChem_data['TestTime'] = EChem_data['TestTime']/3600
    # create a key to get data from df's 
    key = 'Component_'+str(args['component_to_plot'])
    # interpolate EChem data onto the FTIR time axis and store new y-values
    f_EChem = interpolate.interp1d(EChem_data['TestTime'], EChem_data['Volts'])
    xnew = areas['Spec_Time']
    ynew = f_EChem(xnew)
    # initiate a figure
    fig = plt.figure(figsize=(7,10), dpi = 200)
    gs = plt.GridSpec(3,25)
    # define the 4 subplots and their locations
    ax0 = fig.add_subplot(gs[0,0:23])
    ax1 = fig.add_subplot(gs[1,0:23])
    ax2 = fig.add_subplot(gs[2,0:23])
    # ax3 is the colorbar
    ax3 = fig.add_subplot(gs[:,24])
    # store the scatter plot to give as a mappable to the colorbar function
    # areas and maxima are 'normalized' to the third value in the column
    plt1 = ax0.scatter(ynew, areas[key]/areas[key][2], c=xnew, cmap = 'jet', alpha = 0.7, marker = '.')
    ax1.scatter(ynew, centers[key], c=xnew, cmap = 'jet', alpha = 0.7, marker = '.')
    ax2.scatter(ynew, maxima[key]/maxima[key][2], c=xnew, cmap = 'jet', alpha = 0.7, marker = '.')
    plt.colorbar(mappable = plt1, cax = ax3, label = 'Time (hrs)')
    # set axis labels
    ax0.set_ylabel('Area (arb)')
    ax0.set_title(key+' Peak Area vs. Voltage')
    ax1.set_ylabel('Center ($cm^{-1}$)')
    ax1.set_title(key+' Peak Center vs. Voltage')
    ax2.set_ylabel('Height (arb)')
    ax2.set_title(key+' Peak Height vs. Voltage')
    ax2.set_xlabel('Potential (V)')
    # clean up the plot
    fig.tight_layout()
# call the function
AreaCenterMax_vs_Voltage()