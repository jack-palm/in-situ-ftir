# -*- coding: utf-8 -*-
"""
author: Jack Palmer
email: jpalmer1028@gmail.com
"""

import pandas as pd
import matplotlib.pyplot as plt  
import numpy as np
from scipy.spatial import ConvexHull

##############################################################################
################################ USER INPUTS #################################
##############################################################################

args = {
    # Path raw data to be fit
    'raw_file' : 'C:/Users/someuser/folder_with_data/data_for_fit.txt',
    # Path to another spectrum to add to the plot if desired
    'ref_file' : 'C:/Users/someuser/folder_with_data/data_for_reference.txt',
    # Do you want to plot the ref_file? Boolean
    'use_ref' : True,
    # Set limits of the x-axis
    'left_lim' : 2000,
    'right_lim' : 400
}

##############################################################################
##############################################################################
##############################################################################

def BaselineTest(args):
    # import the raw data to be fit
    raw_file = args['raw_file']    
    raw_data = pd.read_csv(raw_file, header = None)
    raw_data = raw_data.sort_values(0, ignore_index = True)
    raw_x = raw_data[0]
    raw_y = raw_data[1]
    # If specified, import the reference spectrum
    if args['use_ref'] == True:
        ref_file = args['ref_file'] #optional
        ref_data = pd.read_csv(ref_file, header = None)
        ref_data = ref_data.sort_values(0, ignore_index = True)
        ref_x = ref_data[0]
        ref_y = ref_data[1]
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
    
    fit_x = raw_x
    # Subtract the baseline
    fit_y = raw_y - rubberband(raw_x, raw_y)
    # Initiate a plot and plot the raw vs. baselined spectrum.
    # The data are normalized to 0<y<1 for ease of comparison
    plt.figure(dpi = 200)
    plt.plot(fit_x, fit_y/max(fit_y), label = 'New Fit')
    plt.plot(raw_x, (raw_y-min(raw_y))/max(raw_y), label = 'Raw Data')
    # If desired, plot the reference spectrum
    if args['use_ref'] == True:
        plt.plot(ref_x, ref_y/max(ref_y), label = 'Reference')
    # Specify plot features    
    plt.xlim(args['left_lim'], args['right_lim'])
    plt.xlabel("Wavenumber ($cm^{-1}$)")
    plt.ylabel("Absorbance (a.u.)")
    plt.title('Test of Convex Hull Baseline')
    plt.legend()
    plt.tight_layout()

BaselineTest(args)
