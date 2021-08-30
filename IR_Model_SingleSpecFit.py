# -*- coding: utf-8 -*-
"""
author: Jack Palmer
email: jpalmer1028@gmail.com
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import matplotlib.pylab as pl
from lmfit.models import GaussianModel
from scipy import stats

##############################################################################
################################ USER INPUTS #################################
##############################################################################

# Define your intial parameters here.  
initial_vals = {'filepath':'C:/Users/someuser/folder_with_data/datafile.txt',
                'Plot_Title' : 'Generic Title',
                'lower_bound' : 400, # Lowest wavenumber of fitting domain
                'upper_bound' : 800, # Highest wavenumber of fitting domain
                'electrode_peaks' : [440, 460, 560], # Wavenumbers of components
                'electrolyte_peaks' : [500, 700, 720], # Wavenumbers of components
                'tolerance' : 10.0, # Amount allowed to deviate from peaks defined above 
                'first_vals' : pd.DataFrame(np.array([[1,440,17], 
                                                    [1,460,14], 
                                                    [1,500,10],
                                                    [1,560,7],
                                                    [1,700,4],
                                                    [1,720,4]]),
                                           columns=['amplitude', 'center', 'sigma'])
                # 'first_vals' is the model's first guess at parameters
}
# Add 'x_peaks' to initial_vals: all peaks added to one list. Sort it.
initial_vals['x_peaks'] = initial_vals['electrode_peaks']+initial_vals['electrolyte_peaks']
initial_vals['x_peaks'].sort()

##############################################################################
##############################################################################
##############################################################################


def single_fit():
    
    """
    "get_insitu_data" extracts the desired region of the specified spectrum
    and outputs the wavenumbers and absorbances as x_fit and y_fit respectively
    """
    def get_data(filepath, lower_bound, upper_bound):
        
        data = pd.read_csv(filepath, header = None,skiprows=1)
        # select out the desired data from the dataframe
        x_vals = list(data.loc[:,0])
        y_vals = list(data.loc[:,1])
        
        # create and populate a list containing rounded x-values
        rounded_x = []
        for i in range(len(x_vals)):
            rounded_x.append(int(np.round(x_vals[i])))
          
        # since wavenumber does not increase by exactly 1, we need to map the 
        # desired bound to a rounded x-value and extract the index. If that 
        # value does not exist, the next closest value is taken.
        if lower_bound in rounded_x == True:
            x1_ind = rounded_x.index(lower_bound)
        else:
            x1_ind = rounded_x.index(lower_bound + 1)
          
        if upper_bound in rounded_x == True:
            x2_ind = rounded_x.index(upper_bound)
        else:
            x2_ind = rounded_x.index(upper_bound - 1)
        
        # slice the desired data down to the specified range and store
        if x2_ind > x1_ind:
            x_fit = x_vals[x1_ind:x2_ind]
            y_fit = y_vals[x1_ind:x2_ind]
        else:
            x_fit = x_vals[x2_ind:x1_ind]
            y_fit = y_vals[x2_ind:x1_ind]
        return x_fit, y_fit

    """
    "get_fit_parameters" is where the magic happens. The lmfit Gaussian
    model is used to fit the data. The first guess at parameters comes from 
    first_vals and y_fit is used as the data to be fit. This function outputs
    the best fit parameters as "best_vals" and the component names.
    
    """
    def get_fit_parameters(x_fit, y_fit, x_peaks, first_vals):
      
        # Initiate the model by adding the first component
        # Define the model parameters using first_vals
        sigma = first_vals.loc[0,'sigma']
        center=first_vals.loc[0,'center']
        A = first_vals.loc[0,'amplitude']
        # Initiate the dict to store the model components
        components = {}
        # Initiate a list to store the component names
        component_names = []
        # Name the component
        prefix = 'Component' + '_' + str(x_peaks[0])
        # Call the GaussianModel
        peak = GaussianModel(prefix=prefix)
        # Set the initial parameter guesses
        pars = peak.make_params(center=center, sigma=sigma, amplitude=A)
        # Define the maximum amount this peak center can wander from its 
        # initial guess.
        pars[prefix+'center'].set(min=center-initial_vals['tolerance'], 
                                  max=center+initial_vals['tolerance']) 
        # All amplitudes must be positive
        pars[prefix+'amplitude'].set(min=0.1)
        pars[prefix+'sigma'].set(min=1,max=20)
        # Add the component and its name to the respective dict and list
        components[prefix] = peak
        component_names.append(prefix)
        # Assign this peak to "mod". This variable will be appended iteratively
        # to create the overall model
        mod = components[component_names[0]]
        
        # If there is more than one peak, the following for loop will exectute
        if len(x_peaks) > 1:
            # This for loop is identical to the process for defining and adding
            # components outlined above. It is now iterative.
            for i in np.arange(1 , len(x_peaks)):
                
                sigma = first_vals.loc[i,'sigma']
                center=first_vals.loc[i,'center']
                A = first_vals.loc[i,'amplitude']
                prefix = 'Component' + '_' + str(x_peaks[i])
                
                peak = GaussianModel(prefix=prefix)
                pars.update(peak.make_params(center=center, sigma=sigma, amplitude=A))
                pars[prefix+'center'].set(min=center-initial_vals['tolerance'],
                                          max=center+initial_vals['tolerance']) 
                pars[prefix+'amplitude'].set(min=0.1)
                pars[prefix+'sigma'].set(min=1,max=20)
                components[prefix] = peak
                component_names.append(prefix)
                mod += components[component_names[i]]
            
        # Exectute the fitting operation and store to "out"
        out = mod.fit(y_fit, pars, x=x_fit, method = 'lbfgsb')
        # Plot the fit using lmfit's built-in plotting function, includes fit
        # residuals
        # out.plot(fig=1)
        # Create an array of zeros to populate a dataframe
        d = np.zeros((len(x_peaks),3))
        # Create a dataframe to store the best fit parameter values
        best_vals = pd.DataFrame(d ,columns = ['amplitude',
                                             'center', 
                                             'sigma'])
        # Populate the dataframe with the best fit values
        for i in range(len(x_peaks)):
            best_vals.loc[i,'amplitude'] = out.best_values[component_names[i] + 'amplitude']
            best_vals.loc[i,'center'] = out.best_values[component_names[i] + 'center']
            best_vals.loc[i,'sigma'] = out.best_values[component_names[i] + 'sigma']
            
        return best_vals, component_names
    
    """
    "plot_components" plots the following onto a single plot: each component,
    the best-fit line, and the raw data. It also stores each component's
    amplitude and area to separate dataframes.
    
    """
    def plot_components(x_fit, y_fit, best_vals, x_peaks, component_names):
        
        # GM is the equation representing the gaussian Model. Given a set 
        # of parameters and x-values, the y-vals are output as "data"
        def GM(amp, mu, sigma):
            data = []
            for x in x_fit:
                y = ((amp)/(sigma*np.sqrt(2*np.pi)))*(np.e**((-(x-mu)**2)/(2*sigma**2)))
                data.append(y)
            return data
        
        # generateY uses GM to output dataframes containing the wavenumbers
        # and absorbances for each component as well as the sum of all
        # components (best-fit line) and stores them to a dictionary "curves"
        def generateY(x_fit, best_vals):
            # initiate the curves dict
            curves = {}
            # prepare data to initiate a dataframe
            d = {'Wavenumber':x_fit,
                 'Abs':0}
            # within the dict "curves", initiate the best_fit df. Each
            # component's absorbance will be added to this df, forming the best
            # fit line.
            curves['Best_Fit'] = pd.DataFrame(d , 
                                              index = range(len(x_fit)), 
                                              columns = ['Wavenumber', 'Abs'])
            # iteratively add each component to the dict "curves"
            for i in range(len(x_peaks)):
                amp = best_vals.loc[i,'amplitude']
                mu = best_vals.loc[i,'center']
                sigma = best_vals.loc[i,'sigma']  
                # add the component to curves using GM and best-fit parameters
                # to produce the absorbance values
                curves[component_names[i]] = pd.DataFrame(list(zip(x_fit,GM(amp, mu, sigma))),
                                                         columns = ['Wavenumber', 'Abs'])
                # add the component to the best fit dataframe 
                curves['Best_Fit']['Abs'] = curves['Best_Fit']['Abs'].add(curves[component_names[i]]['Abs'], fill_value = 0)
            return curves
        
        # Define a function to calculate MSE, RMSE and nRMSE (normalized by the 
        # interquartile range)
        def MSE_RMSE(y_fit, curves):
            
            y_true = list(y_fit)
            y_pred = list(curves['Best_Fit']['Abs'])
            MSE = np.square(np.subtract(y_true,y_pred)).mean()
            RMSE = np.sqrt(MSE)
            IQR = stats.iqr(y_true, interpolation = 'midpoint')
            nRMSE = RMSE/IQR
            
            return [['MSE', 'RMSE', 'nRMSE'],[MSE, RMSE, nRMSE]]
        
        # call generateY to produce the dict. "curves"
        curves = generateY(x_fit, best_vals)
        # Call MSE_RMSE to generate fit scores
        errors = MSE_RMSE(y_fit, curves)
        
        # initiate a figure to plot all the components onto
        plt.figure(figsize=(4.5,4)) 
        plt.figure(dpi = 200)
        plt.xlabel("Wavenumber ($cm^{-1}$)", fontsize=12)
        plt.ylabel("Absorbance (a.u.)", fontsize=12)
        # create a color scheme
        colors = pl.cm.jet(np.linspace(0,1,len(x_peaks)))
        # iteratively add all components to the plot
        for i in range(len(x_peaks)):
            plt.plot(curves[component_names[i]].loc[:,'Wavenumber'], 
                     curves[component_names[i]].loc[:,'Abs'], 
                     label = component_names[i],
                     color = colors[i])
            # shade the area under the curve
            plt.fill_between(curves[component_names[i]].loc[:,'Wavenumber'], 
                             0,
                             curves[component_names[i]].loc[:,'Abs'],
                             alpha=0.3,
                             color = colors[i])
        # add the raw data to the plot
        plt.plot(x_fit, y_fit, linewidth=2, label='Raw Data', color = 'hotpink', alpha = 1)
        # add the best fit to the plot
        plt.plot(curves['Best_Fit']['Wavenumber'],curves['Best_Fit']['Abs'], '--', label='Best Fit', color = 'black', alpha=0.5)
        plt.xlim(initial_vals['upper_bound'], initial_vals['lower_bound'])
        plt.legend(fontsize=5)
        plt.title(initial_vals['Plot_Title'])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        # create a dataframe and populate it with each component's amplitude
        amplitudes = pd.DataFrame(component_names ,columns = ['Components'])
        amplitudes[0] = best_vals['amplitude']
        # create a dataframe and populate it with each component's center
        centers = pd.DataFrame(component_names ,columns = ['Components'])
        centers[0] = best_vals['center']
        # create a dataframe and populate it with each component's sigma
        sigmas = pd.DataFrame(component_names ,columns = ['Components'])
        sigmas[0] = best_vals['sigma']
        # create a dataframe and populate it with each component's area
        areas = pd.DataFrame(component_names ,columns = ['Components'])
        # create a dataframe and populate it with each component's maximum
        maxima = pd.DataFrame(component_names ,columns = ['Components'])
        temp_areas = []
        temp_maxima = []
        for name in component_names:
            temp_areas.append(np.trapz(y = curves[name]['Abs'], 
                                 x = curves[name]['Wavenumber']))
            temp_maxima.append(max(curves[name]['Abs']))
        areas[0] = temp_areas
        maxima[0] = temp_maxima
            
        return curves, amplitudes, centers, areas, sigmas, maxima, errors
    
    # call the functions defined above and store their outputs
    # extract the desired region from the specifed spectrum within insitu_data
    x_fit, y_fit = get_data(initial_vals['filepath'], initial_vals['lower_bound'], initial_vals['upper_bound'])
    # fit the desired data and return the best fit parameter values
    best_vals, component_names = get_fit_parameters(x_fit, y_fit, initial_vals['x_peaks'], initial_vals['first_vals'])
    # plot the fitting result and return the curves, areas, and amplitudes
    curves, amplitudes, centers, areas, sigmas, maxima = plot_components(x_fit, y_fit, best_vals, initial_vals['x_peaks'], component_names)
    
    return best_vals, curves, amplitudes, centers, areas, sigmas, maxima, errors
   
best_vals, curves, amplitudes, centers, areas, sigmas, maxima, errors = single_fit()
