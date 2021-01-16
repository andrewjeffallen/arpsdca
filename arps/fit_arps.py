import pandas 
import numpy as np


from scipy.optimize import curve_fit
from arps.dca_utils import (
      hyperbolic_equation, 
      get_qi,
      plot_dca,
)


def execute_arps(unique_well_APIs_list,prd_time_series, time_interval, days,liquid, b_upper,  di_bound, **kwargs):
    for api_number in unique_well_APIs_list:
        production_time_series=prd_time_series[prd_time_series.API==api_number]
        print("total number of wells in dataset: ",len(production_time_series.API.unique()))

        production_time_series = production_time_series[(production_time_series[liquid].notnull()) & (production_time_series[liquid]>0)]
        print("removed missing and zero values from f'{liquid}' column")

        production_time_series['day'] = production_time_series.groupby(by='API').cumcount()+1

        # convert API & Date to string and liquid to float
        production_time_series=production_time_series.astype({"API": str,'day':int,f'{liquid}':float})

        # Get the highest value of production in the first {days} days of production, to use as qi value

        qi=get_qi(production_time_series, days, liquid, 'day')

        # Hyperbolic curve fit the data to get best fit equation
        popt_hyp, pcov_hyp=curve_fit(hyperbolic_equation, production_time_series['day'], 
                                     production_time_series[liquid],bounds=(0, [qi,b_upper,di_bound]))

        #print('Hyperbolic Fit Curve-fitted Variables: qi='+str(popt_hyp[0])+', b='+str(popt_hyp[1])+', di='+str(popt_hyp[2]))

        #Hyperbolic fit results
        production_time_series.loc[:,'hyperbolic_predicted']=hyperbolic_equation(production_time_series['day'], 
                                  *popt_hyp)
        
        y = production_time_series[liquid]
        y_fit = production_time_series['hyperbolic_predicted']

        # residual sum of squares
        ss_res = np.sum((y - y_fit) ** 2)

        # total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        # r-squared
        r2 = 1 - (ss_res / ss_tot)
        
        print('Hyperbolic Fit Curve-fitted Variables:')
        print('r2='+str(round(r2, 4))+' qi='+str(popt_hyp[0])+', b='+str(popt_hyp[1])+', di='+str(popt_hyp[2]))

        #Declare the x- and y- variables that we want to plot against each other
        y_variables=[f'{liquid}', "hyperbolic_predicted"]
        x_variable='day'

        #Create the plot title
        plot_title='liquid'+' Production for Well API '+str(api_number)

        #Plot the data to visualize the equation fit
        plot_dca(production_time_series, x_variable, y_variables, plot_title)

