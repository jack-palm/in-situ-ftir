# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 14:53:46 2021

@author: jpalmer
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 12:25:39 2021

@author: jpalmer
"""

def initial_fit():
    
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt  
    import matplotlib.pylab as pl
    from lmfit.models import GaussianModel
    
    # Define your intial parameters here.  
    initial_vals = {'folder':'C:/Users/jpalmer/Documents/SCP/Data/FTIR/LNO_1MLiClO4inPC_Data_BLSub',
                    'start' : 110, # Which run do you want to fit? 1 is the first
                    'lower_bound' : 462, # Lowest wavenumber of fitting domain
                    'upper_bound' : 630, # Highest wavenumber of fitting domain
                    'LNO_peaks' : [510,542,570,590], # Wavenumbers of components
                    'LiClO4_PC_peaks' : [583,623], # Wavenumbers of components
                    'tolerance' : 20.0, # Amount allowed to deviate from peaks defined above 
                    'first_vals' : pd.DataFrame(np.array([[1,510,0,17], 
                                                        [1.3,542,0,14], 
                                                        [1,570,0,10],
                                                        [0.5,583,0,7],
                                                        [0.3,590,0,4],
                                                        [1.1,623,0,4]]),
                                               columns=['amplitude', 'center', 'fraction', 'sigma'])
                    # 'first_vals' is the model's first guess at parameters
    
    }
    # Add 'x_peaks' to initial_vals: all peaks added to one list. Sort it.
    initial_vals['x_peaks'] = initial_vals['LNO_peaks']+initial_vals['LiClO4_PC_peaks']
    initial_vals['x_peaks'].sort()
    # store the name of the run for plotting purposes
    run_name = 'Run_' + str(initial_vals['start'])
    
    """
    This function "load_insitu_dataset" reads each file in the specified folder
    and adds each file to a single dataframe.
    """
    def load_insitu_dataset(folder):
        
        #store a list of file names in the folder to files
        files = os.listdir(folder)
        
        #create a list handles and populate with the counter in each file name
        #this will be used to load the files in chronological order
        handles = []
        for name in files:
            handles.append(int(name.split('_')[-3]))
            
        #create a dataframe of filenames and handles to sort the file names
        data = {'Files':files, 
                'Indices':handles} 
        df = pd.DataFrame(data)
        df =df.sort_values(by = ['Indices'])
        df = df.reset_index()
        df = df[['Files','Indices']]
        
        #initiate insitu_data. Add filenames as column  add x coordinates (wavenumber)
        insitu_data = pd.DataFrame(columns = list(df.loc[:,'Files']))
        temp_data = pd.read_csv(folder + '/' + df.loc[0,'Files'], header= None)
        insitu_data['Wavenumber'] = temp_data[0]
        
        #populate insitu_data with all data
        for i in range(len(files)):
            temp_data = pd.read_csv(folder + '/' + df.loc[i,'Files'], header= None)
            insitu_data.iloc[:,i] = temp_data.iloc[:,1]
        
        return insitu_data
    
    """
    "get_insitu_data" extracts the desired region of the specified spectrum
    and outputs the wavenumbers and absorbances as x_fit and y_fit respectively
    """
    def get_insitu_data(insitu_data, run_index, lower_bound, upper_bound):
        
        # select out the desired data from the dataframe
        x_vals = list(insitu_data.loc[:,'Wavenumber'])
        y_vals = list(insitu_data.iloc[:,run_index])
        
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
        x_fit = x_vals[x1_ind:x2_ind]
        y_fit = y_vals[x1_ind:x2_ind]
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
        # call generateY to produce the dict. "curves"
        curves = generateY(x_fit, best_vals)
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
        plt.title(run_name)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        # create a dataframe and populate it with each component's amplitude
        amplitudes = pd.DataFrame(component_names ,columns = ['Components'])
        amplitudes[run_name] = best_vals['amplitude']
        # create a dataframe and populate it with each component's center
        centers = pd.DataFrame(component_names ,columns = ['Components'])
        centers[run_name] = best_vals['center']
        # create a dataframe and populate it with each component's sigma
        sigmas = pd.DataFrame(component_names ,columns = ['Components'])
        sigmas[run_name] = best_vals['sigma']
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
        areas[run_name] = temp_areas
        maxima[run_name] = temp_maxima
            
        return curves, amplitudes, centers, areas, sigmas, maxima
    
    # call the functions defined above and store their outputs
    # load the insitu dataset and store it
    insitu_data = load_insitu_dataset(initial_vals['folder'])
    # extract the desired region from the specifed spectrum within insitu_data
    x_fit, y_fit = get_insitu_data(insitu_data, initial_vals['start'], initial_vals['lower_bound'], initial_vals['upper_bound'])
    # fit the desired data and return the best fit parameter values
    best_vals, component_names = get_fit_parameters(x_fit, y_fit, initial_vals['x_peaks'], initial_vals['first_vals'])
    # plot the fitting result and return the curves, areas, and amplitudes
    curves, amplitudes, centers, areas, sigmas, maxima = plot_components(x_fit, y_fit, best_vals, initial_vals['x_peaks'], component_names)
    
    return insitu_data, best_vals, curves, amplitudes, centers, areas, sigmas, maxima, run_name
   
curves = {} 
insitu_data, best_vals, temp, amplitudes, centers, areas, sigmas, maxima, run_name = initial_fit()
curves[run_name] = temp

"""
The function "fit_all" iterates through the desired range of insitu runs and performs 
the same process as "initial_fit". It selects out the desired spectrum and range to be fit, 
fits the data and plots the best fit, components, and raw data on a single plot. This 
function stores the amplitudes (df) and areas (df) of each component,
the in situ dataframe, the best fit parameter values (df), and the 
xy-coordinates of the best-fit line and each component (dict. of df's)
                                                        
"""

def fit_all(insitu_data, best_vals, curves, amplitudes, areas):
    
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt  
    import matplotlib.pylab as pl  
    from lmfit.models import GaussianModel
    # Define your intial parameters here.  
    initial_vals = {'start' : 110, # Which run do you want to start the fit? 1 is the first. this parameter matters when fitting a select number of spectra
                    'step' : 1, # At what interval should spectra be selected? 1 is recommended because of peak shifting
                    'amount' : 177, # How many total spectra do you want to fit? 
                    # 'all' or 'All' will fit all starting at the first spectrum, 
                    # while entering an integer will fit that many spectra 
                    # starting at the specified spectrum
                    'lower_bound' : 462, # Lowest wavenumber of fitting domain
                    'upper_bound' : 630,  # Highest wavenumber of fitting domain
                    'LNO_peaks' : [510,542,570,590],# Wavenumbers of components
                    'LiClO4_PC_peaks' : [583,623],# Wavenumbers of components
                    'tolerance' : 20.0    
    }
    # Add 'x_peaks' to initial_vals: all peaks added to one list. Sort it.
    initial_vals['x_peaks'] = initial_vals['LNO_peaks']+initial_vals['LiClO4_PC_peaks']
    initial_vals['x_peaks'].sort()
    
    """
    This function "load_insitu_dataset" reads each file in the specified folder
    and adds each file to a single dataframe.
    """
    def get_insitu_data(insitu_data, run_index, lower_bound, upper_bound):
        
        #select out the desired data from the dataframe
        x_vals = list(insitu_data.loc[:,'Wavenumber'])
        y_vals = list(insitu_data.iloc[:,run_index])
        
        #create and populate a list containing rounded x-values
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
        x_fit = x_vals[x1_ind:x2_ind]
        y_fit = y_vals[x1_ind:x2_ind]
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
        out = mod.fit(y_fit, pars, x=x_fit, method ='lbfgsb')
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
    def plot_components(x_fit, y_fit, best_vals, x_peaks, component_names, run_number):
        
        # GM is the equation representing the Gaussian Model. Given a set 
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
        # call generateY to produce the dict. "curves"
        curves = generateY(x_fit, best_vals)
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
                     color=colors[i])
            # shade the area under the curve
            plt.fill_between(curves[component_names[i]].loc[:,'Wavenumber'], 
                             0,
                             curves[component_names[i]].loc[:,'Abs'],
                             alpha=0.3,
                             color=colors[i])        
        # add the raw data to the plot
        plt.plot(x_fit, y_fit, linewidth=2, label='Raw Data',color = 'hotpink')
        # add the best fit to the plot
        plt.plot(curves['Best_Fit']['Wavenumber'],curves['Best_Fit']['Abs'], '--', label='Best Fit',alpha = 0.5,color = 'black')
        plt.xlim(initial_vals['upper_bound'], initial_vals['lower_bound'])
        plt.title('Run_'+ str(run_number))
        plt.legend(fontsize=5)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.show()
        # store the amplitudes to be output from the function
        best_amps = best_vals['amplitude']
        # store the amplitudes to be output from the function
        best_centers = best_vals['center']
        # store the amplitudes to be output from the function
        best_sigmas = best_vals['sigma']
        # initiate a list to store areas
        best_areas = []
        # initiate a list to store maxima
        best_maxima = []
        # append areas to the list
        for name in component_names:
            best_areas.append(np.trapz(y = curves[name]['Abs'], 
                                 x = curves[name]['Wavenumber']))
            best_maxima.append(max(curves[name]['Abs']))
            
        return curves, best_amps, best_centers, best_areas, best_sigmas, best_maxima
    # iterate over the desired range of the insitu dataset
    # if all spectra are desired to be fit, this will execute
    if initial_vals['amount'] == 'all' or initial_vals['amount'] == 'All':
        for i in range(len(insitu_data.columns)-2):
            x_fit, y_fit = get_insitu_data(insitu_data, 
                                           1+i, 
                                           initial_vals['lower_bound'], 
                                           initial_vals['upper_bound'])
            best_vals, component_names = get_fit_parameters(x_fit, y_fit, initial_vals['x_peaks'], best_vals)
            curves['Run_'+str(i+2)], amplitudes['Run_'+str(i+2)], centers['Run_'+str(i+2)], areas['Run_'+str(i+2)], sigmas['Run_'+str(i+2)], maxima['Run_'+str(i+2)] = plot_components(x_fit, y_fit, best_vals, initial_vals['x_peaks'], component_names, i+2)
    # if a select range specified in the initial_vals dict is desired, this
    # will execute
    else:
        for i in range(initial_vals['amount']):
            x_fit, y_fit = get_insitu_data(insitu_data, 
                                           initial_vals['start']+(i*initial_vals['step']), 
                                           initial_vals['lower_bound'], 
                                           initial_vals['upper_bound'])
            best_vals, component_names = get_fit_parameters(x_fit, y_fit, initial_vals['x_peaks'], best_vals)
            curves['Run_'+str(initial_vals['start']+(i*initial_vals['step'])+1)], amplitudes['Run_'+str(initial_vals['start']+(i*initial_vals['step'])+1)], centers['Run_'+str(initial_vals['start']+(i*initial_vals['step'])+1)], areas['Run_'+str(initial_vals['start']+(i*initial_vals['step'])+1)], sigmas['Run_'+str(initial_vals['start']+(i*initial_vals['step'])+1)], maxima['Run_'+str(initial_vals['start']+(i*initial_vals['step'])+1)] = plot_components(x_fit, y_fit, best_vals, initial_vals['x_peaks'], component_names, initial_vals['start']+(i*initial_vals['step'])+1)
            
    return curves, areas, centers, amplitudes, sigmas

# call the function
fit_all(insitu_data, best_vals, curves, amplitudes, areas)
    
"""
"data_export" takes a user defined directory, creates a new folder within that
directory, and populates that folder with the amplitudes, areas, centers,
insitu dataset, and curves for the fit.

"""

def data_export(folder, export): # 'folder' is the desired path for export
                                 # If 'export' is True, this function executes
    import os
    import numpy as np
    import pandas as pd
    from datetime import datetime
    
    if export == True:
        # store the current date and time to the variable 'now' and use it 
        # in the name for the new directory
        now = datetime.now()
        now = now.strftime("%Y_%m_%d_%H_%M_%S")
        path = folder + '/FittingOutput_'+ now
        # create the directory
        os.mkdir(path)
        # store amplitudes, areas, centers, and the entire in situ dataset
        # to individual .csv files under the new directory
        amplitudes.to_csv(path + '/amplitudes.csv', index = None)
        areas.to_csv(path + '/areas.csv', index = None)
        centers.to_csv(path +'/centers.csv', index = None)
        maxima.to_csv(path +'/maxima.csv', index = None)
        sigmas.to_csv(path +'/sigmas.csv', index = None)
        insitu_data.to_csv(path+'/insitu_dataset.csv', index = None)
        # save curves dict to a .npy file for easy loading later
        np.save(path + '/curves_dict.npy', curves)
        # create a folder to save the curves under the new directory
        curve_path = path + '/curves'
        os.mkdir(curve_path)
        # store the keys of curves dict to k
        k = list(curves.keys())
        # iterate over the keys
        for key in k:
            # store the next level of keys to kk
            kk = list(curves[key].keys())
            # create a temporary df using kk as the columns
            temp = pd.DataFrame(columns = kk)
            # add the wavenumbers (x-vals) to the df 'temp'
            temp['Wavenumber'] = curves[key][kk[0]]['Wavenumber']
            # iterate over kk and add absorbances to the df 'temp'
            for key2 in kk:
                temp[key2] = curves[key][key2]['Abs']
            # once all components are added, save the df to the curves directory
            temp.to_csv(curve_path + '/'+key+'.csv', index = None)
            
# call the function. Unless the second arg is 'True', this function will not execute
data_export('C:/Users/jpalmer/Documents/SCP/Data/FTIR/temp', True)


