# -*- coding: utf-8 -*-
"""
author: Jack Palmer
email: jpalmer1028@gmail.com
"""

"""
An iterative plotting script to plot mulitple curves from the raw data onto 
multiple plots. The arguments do the following:
    
    filepath: the absolute path of the insitu_dataset
    start: define the spectrum number to start at on the plot
    step: the interval between spectra
    amount: the number of spectra to put on one plot
    
The function is called within a for loop to create multiple plots

"""
def RawMultiPlot(filepath, start, step, amount):
    
    import matplotlib.pyplot as plt
    import matplotlib.pylab as pl
    import pandas as pd
    import numpy as np
    # import the insitu dataset as output from the main fitting script
    insitu_data = pd.read_csv(filepath)
    # define plot features
    plt.figure(dpi = 200)
    plt.xlim(2000, 450)
    plt.ylim(-0.01, 0.55)
    plt.title('In Situ FTIR '+str(start+1)+'-'+str(start+(step*amount)), fontsize = 16)
    plt.xlabel("Wavenumber ($cm^{-1}$)", fontsize=12)
    plt.ylabel("Absorbance (a.u.)", fontsize=12)
    colors = pl.cm.jet(np.linspace(0,1,amount))
    # iterate over range(amount) to add data to the plot
    for i in range(amount):
        plt.plot(insitu_data['Wavenumber'], insitu_data.iloc[:,(start+(step*i))], 
                 label = str(start+(step*i)+1), 
                 color = colors[i],
                 alpha = 0.65)
    
    plt.legend(fontsize = 7)
    
# iterate over the desired number of plots and call the function.
for i in range(5):    
    RawMultiPlot('C:\\Users\LMEYER\Desktop/test_spec.csv', (i*2), 1, 3)

#Data must be in columns of absorbance so
#Wavenumber Absorbance 1 Absorbance 2 Absorbance 3
#Range is the number of plots
#(i*x) x is is the step size between the first spectra listed shown in the plots
# second number is the starting spectrum
#third number is the amount of spectra per plot
