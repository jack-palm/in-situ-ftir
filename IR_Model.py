
from selectors import EpollSelector
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_point_clicker import clicker
import os
from natsort import natsorted
import matplotlib.pylab as pl
from lmfit.models import GaussianModel
from scipy.signal import savgol_filter as sgf
from scipy import stats
from insitu_ftir.helpers import *
# must use %matplotlib widget for interactive plotting

class IR_Model:

    def __init__(self, initial_vals):

        self.data_dir = initial_vals['folder']
        self.export_folder = initial_vals['export_folder']
        self.start_ind = initial_vals['start']
        self.step = initial_vals['step']
        self.amount = initial_vals['amount']
        self.lower_bound = initial_vals['lower_bound']
        self.upper_bound = initial_vals['upper_bound']

        self.guess_vals = initial_vals['first_vals']
        self.guess_vals['centers'] = initial_vals['first_vals']['center']
        self.guess_vals = self.guess_vals.set_index('centers')
        self.guess_vals = self.guess_vals.sort_index()

        self.x_peaks = list(self.guess_vals['center'])

        self.fixed_peaks = self.guess_vals[self.guess_vals['mobile'] == False]
        self.mobile_peaks = self.guess_vals[self.guess_vals['mobile'] == True]
        self.tolerance = initial_vals['tolerance']
        self.min_amplitude = initial_vals['min_amplitude']
        self.max_amplitude = initial_vals['max_amplitude']
        self.min_sigma = initial_vals['min_sigma']
        self.max_sigma = initial_vals['max_sigma']
        self.run_ind = f'Run_{initial_vals["start"]}'

        self.load_data()
        self.initial_scan = list(self.data.items())[0][1]

    def load_data(self, data_dir == None):
        """
        This function will load the data and set _data_loaded == True
        """
        data_dir = os.getcwd()
        fids = [os.path.join(data_dir, p) for p in os.listdir() if ".txt" in p]
        fids = natsorted(fids)

        def get_nums(fids):
            
            nums = [f.replace('_', '.').split('.') for f in fids]
            nums = natsorted(np.concatenate(nums))
            run_nums = []
            for n in nums:
                try:
                    run_nums.append(int(n))
                except:
                    pass

            return run_nums

        self.numbers = get_nums(fids)
        if len(self.numbers) == len(fids):
            self.data = {num: pd.read_csv(fid, header = None) for num,fid in zip(self.numbers,fids)}
        else:
            self.data = {fid: pd.read_csv(fid, header = None) for fid in fids}

        self._data_loaded = True


    def set_peaks(self):
        """ 
        This function will plot the first spectrum in a fitting sequence and 
        allow you to set the initial peak positions and intensities.
        """
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot()
        plt.show()
        klicker = clicker(ax, ["event"], markers=["x"])
        
        return klicker # then use klicker.get_positions() to return the positions

    def set_bounds(self):
        """ 
        This function will plot the first spectrum in a fitting sequence and 
        allow you to set the initial peak positions and intensities.
        """
        fig, ax = plt.subplots(constrained_layout=True)

        ax.plot(self.initial_scan[0], self.initial_scan[1])
        klicker = clicker(ax, ["event"], markers=["x"])
        plt.show()
        input("Please select upper and lower bounds for fitting")
    
        
        return klicker.get_positions() # then use klicker.get_positions() to return the positions

    def get_insitu_data(self):

        x = np.array(self.data[self.run_index][0])
        y = np.array(self.data[self.run_index][1])

        x1_ind = find_nearest(x, self.lower_bound)
        x2_ind = find_nearest(x, self.upper_bound)

        if x2_ind > x1_ind:
            self.x_fit = x[x1_ind:x2_ind]
            self.y_fit = y[x1_ind:x2_ind]
        else:
            self.x_fit = x[x2_ind:x1_ind]
            self.y_fit = y[x2_ind:x1_ind]


    def get_fit_parameters(self):
        
        amp, center, sigma, mobile = self.guess_vals.iloc[0,:]
       
        components = {}
        prefix = f'Component_{center}'
        peak = GaussianModel(prefix = prefix)
        pars = peak.make_params(
            center = center,
            sigma = sigma,
            amplitude = amp
        )

        if mobile == False:
            pars[f'{prefix}center'].set(
                min = center-(self.tolerance/100),
                max = center+(self.tolerance/100)
            )
        else: 
            pars[f'{prefix}center'].set(
                min = center-self.tolerance,
                max = center+self.tolerance
            )

        pars[f'{prefix}amplitude'].set(
            min = self.min_amplitude,
            max = self.max_amplitude
        )       
        pars[f'{prefix}sigma'].set(
            min = self.min_sigma,
            max = self.max_sigma
        )
        components[prefix] = peak
        mod = peak

        for _, (amp, center, sigma, mobile) in self.guess_vals[1:].iterrows():

            prefix = f'Component_{center}'
            peak = GaussianModel(prefix = prefix)
            pars = peak.make_params(
                center = center,
                sigma = sigma,
                amplitude = amp
            )

            if mobile == False:
                pars[f'{prefix}center'].set(
                    min = center-(self.tolerance/100),
                    max = center+(self.tolerance/100)
                )
            else: 
                pars[f'{prefix}center'].set(
                    min = center-self.tolerance,
                    max = center+self.tolerance
                )

            pars[f'{prefix}amplitude'].set(
                min = self.min_amplitude,
                max = self.max_amplitude
            )       
            pars[f'{prefix}sigma'].set(
                min = self.min_sigma,
                max = self.max_sigma
            )
            components[prefix] = peak
            mod += peak
    
        out = mod.fit(self.y_fit, pars, self.x_fit, method = 'lbfgsb')



