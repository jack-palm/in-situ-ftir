# -*- coding: utf-8 -*-
"""
Created on Tue May 11 08:10:46 2021

@author: jpalmer
"""
def AreaCenterMax_vs_Voltage():
    
    import pandas as pd
    import matplotlib.pyplot as plt  
    from scipy import interpolate
    # from tkinter import filedialog as fd
    
    # Enter the paths for the fit results and the corresponding EChem data
    EChem_file = 'C:/Users/jpalmer/Documents/SCP/Data/FTIR/LNO_1MLiClO4inPC_Data_BLSub/BTV_LNO_1MLiClO4inPC_JH17 - 095.txt'
    Fit_folder = 'C:/Users/jpalmer/Documents/SCP/Data/FTIR/temp/FittingOutput_2021_05_14_11_29_21'   
    # Enter a single integer corresponding tp the component you would like to plot
    component_to_plot = 591 # 503, 542, 560, 568, 590, 622
    # EChem_file = fd.askopenfilename()           
    # Fit_folder = fd.askdirectory()
    
    # Import the echem data as and make the time column numeric
    EChem_data = pd.read_csv(EChem_file,skiprows = 2,sep='\t')   
    EChem_data['TestTime'] = EChem_data['TestTime'].str.replace(",","").astype(float)
    # Import the fit results
    areas = pd.read_csv(Fit_folder + '/areas.csv')
    centers = pd.read_csv(Fit_folder + '/centers.csv')
    maxima = pd.read_csv(Fit_folder + '/maxima.csv')
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
    # is taken every 10 minutes and is started at t = 0. EChem data is assumed
    # be reported in minutes.
    areas['Spec_Time'] = ((areas['Spec_Time']*10) - 10)/60
    centers['Spec_Time'] = ((centers['Spec_Time']*10) - 10)/60
    maxima['Spec_Time'] = ((maxima['Spec_Time']*10) - 10)/60
    EChem_data['TestTime'] = EChem_data['TestTime']/3600
    # create a key to get data from df's 
    key = 'Component_'+str(component_to_plot)
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