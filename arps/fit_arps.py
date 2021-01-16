import pandas 
import numpy as np
import sys


from scipy.optimize import curve_fit
from arps.dca_utils import (
      hyperbolic_equation, 
      get_qi,
      plot_dca,
)


def execute_arps(well_list,prod_data, days,liquid, b_upper,  di_bound, **kwargs):
      
      well_list = prod_data.API.unique()
      
      for api_number in well_list:
            
            prod_ts=prod_data[prod_data.API==api_number]
              print("total number of wells in dataset: ",len(prod_ts.API.unique()))

              prod_ts = prod_time_series[(prod_ts[liquid].notnull()) & (prod_ts[liquid]>0)]
              print("removed missing and zero values from f'{liquid}' column")

              prod_ts['day'] = prod_ts.groupby(by='API').cumcount()+1

              # convert API & Date to string and liquid to float
              prod_ts=prod_ts.astype({"API": str,'day':int,f'{liquid}':float})

              # Get the highest value of production in the first {days} days of production, to use as qi value

              qi=get_qi(prod_ts, days, liquid, 'day')

              # Hyperbolic curve fit the data to get best fit equation
              popt_hyp, pcov_hyp=curve_fit(hyperbolic_equation, prod_ts['day'], 
                                           prod_ts[liquid],bounds=(0, [qi,b_upper,di_bound]))

              #print('Hyperbolic Fit Curve-fitted Variables: qi='+str(popt_hyp[0])+', b='+str(popt_hyp[1])+', di='+str(popt_hyp[2]))

              #Hyperbolic fit results
              prod_ts.loc[:,'hyperbolic_predicted']=hyperbolic_equation(prod_ts['day'], 
                                        *popt_hyp)

              y = prod_ts[liquid]
              y_fit = prod_ts['hyperbolic_predicted']

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
              plot_dca(prod_ts, x_variable, y_variables, plot_title)

if __name__ == "__main__":
      
      well_list = sys.argv[1]
      prod_data = sys.argv[2]
      days = sys.argv[3]
      liquid = sys.argv[4]
      b_upper = sys.argv[6]
      di_bound = sys.argv[7]
      
      return_code = execute_arps(well_list,prod_data, days,liquid, b_upper,  di_bound)
      
      if return_code == 0:
            sys.exit(0)
      else:
            raise SystemError(f"Error {return_code}")
