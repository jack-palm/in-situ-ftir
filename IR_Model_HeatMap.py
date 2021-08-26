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
    'Plot_Title' : 'Test Map',
    # Fit data export folder
    'Folder_Path' : 'C:/Users/someuser/folder_with_data',
    # Which components to plot. Must match column names in 'curves'
    'Components' : [623] #503, 515, 544, 623, 567, 580, 591
}

##############################################################################
##############################################################################
##############################################################################

def heatmap(args):
    
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt  
    from natsort import natsorted #3rd party library for natural sorting
    
    # Store all file names in the 'curves' folder
    files = os.listdir(args['Folder_Path']+'/curves')
    
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
        
    # Extract all data from the curves folder and store to the dict 'curves'
    curves = {}
    for i in range(len(files)):
        curves[keys[i]] = pd.read_csv(args['Folder_Path'] + '/curves/' + files[i])
        
    # Create a df with counters as the column names
    data = pd.DataFrame(columns=counter)
    
    # Add wavenumbers to the df
    data['Wavenumber'] = curves[keys[0]]['Wavenumber']
    
    # Replace all nan's with zeros
    data = data.fillna(0)
    
    # For each spectrum, sum the desired components and store to the df
    for i in range(len(keys)):
        for x in args['Components']:
            data[counter[i]] = data[counter[i]] + curves[keys[i]]['Component_'+str(x)]

    # Round the wavenumbers in anticipation for making them the indices of the df
    data['Wavenumber'] = data['Wavenumber'].apply(lambda x: np.round(x))
    
    # Set wavenumbers as indicies
    data = data.set_index('Wavenumber')
    
    # To prepare the df for plotting, do the following transformations:
        
    # Transpose the df
    data = data.transpose() 
    
    # Flip the order of the columns
    columns = data.columns.tolist()
    columns = columns[::-1]
    data = data[columns]
    
    # Flip the order of the rows
    data = data[::-1]

    # Store column names (wavenumbers) to x
    x = list(data.columns)
    
    # Store the indices (spectrum number) to y
    y = []
    for i in range(len(data.index)):
        y.append(int(list(data.index)[i]))
    
    # Create a figure and plot the data
    plt.figure(figsize=(5,5), dpi = 200) 
    plt.pcolormesh(x, y,data, cmap = 'viridis', shading = 'nearest')
    plt.xlabel("Wavenumber ($cm^{-1}$)", fontsize=12)
    plt.ylabel("Spectrum Number", fontsize=12)
    plt.title(args['Plot_Title'])
    
    # Invert the x-axis to show higher energy wavenumbers on the right 
    ax = plt.gca()
    ax.invert_xaxis()
    
# call the function
heatmap(args)

