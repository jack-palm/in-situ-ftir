# -*- coding: utf-8 -*-
"""
author: Jack Palmer
email: jpalmer1028@gmail.com
"""

import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull
from natsort import natsorted
import os

##############################################################################
################################ USER INPUTS #################################
##############################################################################

args = {
    # Path to raw data
    'folder' : 'C:/demo/test/',
    # 'all' or 'All' will baseline the entire spectrum
    # a list of two values low to high will baseline over that range
    'region' : 'all'
}

##############################################################################
##############################################################################
##############################################################################

def BaselineCorrect(args):
    
    # Define the function that computes the convex hull around the spectrum  
    def rubberband(x, y):
        # Find the convex hull
        v = ConvexHull(np.array(list(zip(x, y)))).vertices
        # Rotate convex hull vertices until they start from the lowest one
        v = np.roll(v, -v.argmin())
        # Leave only the ascending part
        v = v[:v.argmax()]
    
        # Create baseline using linear interpolation between vertices
        return np.interp(x, x[v], y[v])
    # make the folder that will contain the baselined data
    new_folder = args['folder']+'_BLSub'
    os.mkdir(new_folder)
    # load files and sort by counter using natsorted()
    files = os.listdir(args['folder'])
    files = natsorted(files)
    # iterate over the file names
    for file in files:
        # define your save and import file paths
        import_path = args['folder'] +'/'+ file
        file_save_name = file.replace('.','_')
        file_save_name = file_save_name.replace('_txt','.txt')
        save_path = new_folder + '/' + file_save_name
        #  load data and sort based on wavenumber
        raw_data = pd.read_csv(import_path, header = None)
        raw_data = raw_data.sort_values(0, ignore_index = True)
        #  if a baselining of the entire spectrum is desired, this will execute
        if args['region'] == 'all' or args['region'] == 'All':
            x = raw_data[0]
            raw_y = raw_data[1]
        # if you only want a specific region, this will execute
        elif type(args['region']) == list:
            # since wavenumber does not increase by exactly 1, we need to map the 
            # desired bound to a rounded x-value and extract the index. If that 
            # value does not exist, the next closest value is taken.
            def get_data(raw_data, lower_bound, upper_bound):       
                #select out the desired data from the dataframe
                x_vals = list(raw_data[0])
                y_vals = list(raw_data[1])                
                #create and populate a list containing rounded x-values
                rounded_x = []
                for i in range(len(x_vals)):
                    rounded_x.append(int(np.round(x_vals[i])))
                # Find the index of the lower bound. Once it is found, break
                for i in range(4):
                    if (lower_bound + i in rounded_x) == True:
                        x1_ind = rounded_x.index(lower_bound + i)
                        break
                # Find the index of the upper bound. Once it is found, break
                for i in range(4):
                    if (upper_bound - i in rounded_x) == True:
                        x2_ind = rounded_x.index(upper_bound - i)
                        break                        
                # slice the desired data down to the specified range and store
                x_fit = pd.Series(x_vals[x1_ind:x2_ind])
                y_fit = pd.Series(y_vals[x1_ind:x2_ind])
                return x_fit, y_fit
            # Call the function
            x, raw_y = get_data(raw_data, args['region'][0], args['region'][1])           
        # Run the rubberband function and subtarct the baseline from the spectrum
        fit_y = raw_y - rubberband(x, raw_y)
        # Make the data into a dataframe and save it
        export = pd.DataFrame(zip(x,fit_y))
        export.to_csv(save_path, index = None, header = None)
 # Call the function
BaselineCorrect(args)















