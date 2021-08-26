# -*- coding: utf-8 -*-
"""
author: Jack Palmer
email: jpalmer1028@gmail.com
"""

##############################################################################
################################ USER INPUTS #################################
##############################################################################

args = {# The path of the folder containing the desired fitting output
        'folder' : 'C:/Users/someuser/folder_with_data',
        # A list of the wavenumbers of components you want to plot
        'components' : [500] #500, 510, 540, 620, 560, 580, 590
}

##############################################################################
##############################################################################
##############################################################################

"""
'plot_fit_results' plots the areas, centers, and maxima versus spectrum number
for the desired components from the fit.

"""

def plot_fit_results(components_to_plot, folder):
    
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pl
    import numpy as np
    import pandas as pd
    
    areas = pd.read_csv(folder + '/areas.csv')
    centers = pd.read_csv(folder + '/centers.csv')
    maxima = pd.read_csv(folder + '/maxima.csv')
    # initiate the figure and add a grid for subplots
    fig = plt.figure(figsize=(15,5), dpi = 200)
    gs = plt.GridSpec(1,3)
    # define the two subplots and their locations
    ax0 = fig.add_subplot(gs[0,0])
    ax1 = fig.add_subplot(gs[0,1])
    ax2 = fig.add_subplot(gs[0,2])
    # deifne a color scheme for the component traces
    colors = pl.cm.jet(np.linspace(0,1,len(components_to_plot)))
    # iterate over the defined components, extract, and plot the areas
    for number in components_to_plot:
        key = 'Component_'+str(number)
        # get the index of the component
        x_ind = list(areas['Components']).index(key)
        # extract the data from areas
        data = pd.DataFrame(areas.iloc[x_ind, :])
        # extract the spectrum number from the index
        ind = list(data.index)[1:]
        for i in range(len(ind)):
            ind[i] = int(ind[i].split(sep = '_')[-1])
        # clean up the dataframe
        data = data.reset_index()
        data = data.drop(columns = 'index')
        data.columns = data.iloc[0]
        data = data.drop(data.index[0])
        data = data.reset_index()
        data = data.drop(columns = 'index')
        data['Run_Number'] = ind
        # add the data to the plot
        ax0.plot(data['Run_Number'], data[key], label = key, color = colors[components_to_plot.index(number)])
    # define the axis titles and legend size
    ax0.legend(fontsize = 8)
    ax0.set_title('Areas')
    ax0.set_xlabel('Spectrum Number', fontsize = 10)
    ax0.set_ylabel('Area (arb.)', fontsize = 10)
    # do the same process as above, but for centers of components
    for number in components_to_plot:
        key = 'Component_'+str(number)
        x_ind = list(centers['Components']).index(key)
        data = pd.DataFrame(centers.iloc[x_ind, :])
        ind = list(data.index)[1:]
        for i in range(len(ind)):
            ind[i] = int(ind[i].split(sep = '_')[-1])
        data = data.reset_index()
        data = data.drop(columns = 'index')
        data.columns = data.iloc[0]
        data = data.drop(data.index[0])
        data = data.reset_index()
        data = data.drop(columns = 'index')
        data['Run_Number'] = ind
        ax1.plot(data['Run_Number'], data[key], label = key, color = colors[components_to_plot.index(number)])
    # define the axis titles and legend size
    ax1.legend(fontsize = 8)
    ax1.set_title('Centers')
    ax1.set_xlabel('Spectrum Number', fontsize = 10)
    ax1.set_ylabel('Center ($cm^{-1}$)', fontsize = 10)
    
    for number in components_to_plot:
        key = 'Component_'+str(number)
        # get the index of the component
        x_ind = list(maxima['Components']).index(key)
        # extract the data from areas
        data = pd.DataFrame(maxima.iloc[x_ind, :])
        # extract the spectrum number from the index
        ind = list(data.index)[1:]
        for i in range(len(ind)):
            ind[i] = int(ind[i].split(sep = '_')[-1])
        # clean up the dataframe
        data = data.reset_index()
        data = data.drop(columns = 'index')
        data.columns = data.iloc[0]
        data = data.drop(data.index[0])
        data = data.reset_index()
        data = data.drop(columns = 'index')
        data['Run_Number'] = ind
        # add the data to the plot
        ax2.plot(data['Run_Number'], data[key], label = key, color = colors[components_to_plot.index(number)])
    # define the axis titles and legend size
    ax2.legend(fontsize = 8)
    ax2.set_title('Maxima')
    ax2.set_xlabel('Spectrum Number', fontsize = 10)
    ax2.set_ylabel('Intensity (arb.)', fontsize = 10)

# call the function
plot_fit_results(args['components'], args['folder'])

