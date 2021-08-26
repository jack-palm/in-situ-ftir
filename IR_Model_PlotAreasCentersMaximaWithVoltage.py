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
    'components_to_plot' : [500, 510, 540, 560, 580, 590, 620], # 500, 510, 540, 620, 560, 580, 590
    'spectrum_interval' : 10 # interval in minutes at which FTIR spectra were taken
}

##############################################################################
##############################################################################
##############################################################################

def AreaCenterMax_with_Voltage():
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt  
    import matplotlib.pylab as pl
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
    # store component names to keys
    keys = []
    for num in args['components_to_plot']:
        keys.append('Component_'+str(num))
    # initiate the figure
    fig = plt.figure(figsize=(15,5), dpi = 200)
    gs = plt.GridSpec(1,3)
    # define the three subplots and their locations
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[0,2])
    # deifne a color scheme for the component traces
    colors = pl.cm.jet(np.linspace(0,1,len(args['components_to_plot'])))
    # plot areas, centers, and maxima. Normalize areas and maxima
    ax0.set_xlabel('Time (hrs)')
    ax0.set_ylabel('Peak Area (arb)')
    ax0.set_title('Areas')
    for i in range(len(keys)):
        ax0.plot(areas['Spec_Time'], 
                 areas[keys[i]]/areas[keys[i]][2],
                 label = keys[i],
                 color = colors[i])
    ax0.legend()
    ax0.set_xlim(min(areas['Spec_Time']), max(areas['Spec_Time']))
    
    ax1.set_xlabel('Time (hrs)')
    ax1.set_ylabel('Center ($cm^{-1}$)')
    ax1.set_title('Centers')
    for i in range(len(keys)):
        ax1.plot(centers['Spec_Time'], 
                 centers[keys[i]],
                 label = keys[i],
                 color = colors[i])
    ax1.legend()
    ax1.set_xlim(min(centers['Spec_Time']), max(centers['Spec_Time']))

    ax2.set_xlabel('Time (hrs)')
    ax2.set_ylabel('Peak Maxima (arb)')
    ax2.set_title('Maxima')
    for i in range(len(keys)):
        ax2.plot(maxima['Spec_Time'], 
                 maxima[keys[i]]/maxima[keys[i]][2],
                 label = keys[i],
                 color = colors[i])
    ax2.legend()
    ax2.set_xlim(min(maxima['Spec_Time']), max(maxima['Spec_Time']))
 
    # add voltage profiles to each plot
    ax3 = ax0.twinx()
    ax3.plot(EChem_data['TestTime'], 
             EChem_data['Volts'],
             color = 'black',
             alpha = 0.5)
    ax3.set_ylabel('Potential (V)')
    ax4 = ax1.twinx()
    ax4.plot(EChem_data['TestTime'], 
             EChem_data['Volts'],
             color = 'black',
             alpha = 0.5)
    ax4.set_ylabel('Potential (V)')
    ax5 = ax2.twinx()
    ax5.plot(EChem_data['TestTime'], 
             EChem_data['Volts'],
             color = 'black',
             alpha = 0.5)
    ax5.set_ylabel('Potential (V)')
    # clean up the figure
    fig.tight_layout()
    
    return areas, centers, maxima, EChem_data
# call the function
areas, centers, maxima, EChem_data = AreaCenterMax_with_Voltage()