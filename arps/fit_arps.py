import pandas 
import numpy


from scipy.optimize import curve_fit
from dca import (
      hyperbolic_equation, 
      get_min_or_max_value_in_column_by_group,
      get_max_initial_production,
      plot_actual_vs_predicted_by_equations,
      remove_nan_and_zeroes_from_columns
)


def execute_arps(unique_well_APIs_list,prd_time_series, days,liquid, b_upper,  di_bound, **kwargs):
      """
      prd_time_series: production data in .csv format with fields API, liquid_type, date
      liquid: the production time series to fit Arps curve on (oil, gas, water, etc)
      di_bound: upper bound value for di
      b_bound: upper bound value for b
      """
      
      for api_number in unique_well_APIs_list:
            
            # read in prd_time_series.csv as a dataframe
            df = prd_time_series
            
            # remove nulls and zeroes from the production time series
            df = remove_nan_and_zeroes_from_columns(df,liquid)
            
            #create running 'day' column
            df['day'] = df.groupby(by='API').cumcount()+1
            
            # convert API & Date to string and liquid to float
            df=df.astype({"API": str,'day':str,f'{liquid}':float})
            
            # remove nulls and zeroes from the production time series
           
            
            #Subset the dataframe by API Number
            production_time_series=df[df.API==api_number]
            
            
            # Get the highest value of production in the first {days} days of production, to use as qi value

            qi=get_max_initial_production(production_time_series, days, liquid, 'day')

            # Hyperbolic curve fit the data to get best fit equation
            popt_hyp, pcov_hyp=curve_fit(hyperbolic_equation, production_time_series['day'], 
                                         production_time_series[liquid],bounds=(0, [qi,b_upper,di_bound]))
            
            print('Hyperbolic Fit Curve-fitted Variables: qi='+str(popt_hyp[0])+', b='+str(popt_hyp[1])+', di='+str(popt_hyp[2]))

            #Hyperbolic fit results
            production_time_series.loc[:,'hyperbolic_predicted']=hyperbolic_equation(production_time_series['day'], 
                                      *popt_hyp)

            #Declare the x- and y- variables that we want to plot against each other
            y_variables=[f'{liquid}', "hyperbolic_predicted"]
            x_variable='day'
            
            #Create the plot title
            plot_title='liquid'+' Production for Well API '+str(api_number) 
            
            #Plot the data to visualize the equation fit
            plot_actual_vs_predicted_by_equations(production_time_series, x_variable, y_variables, plot_title)
