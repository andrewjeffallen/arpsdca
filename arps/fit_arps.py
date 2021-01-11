import pandas
import 
from dca import (
hyperbolic_equation, 
get_min_or_max_value_in_column_by_group,
get_max_initial_production,
plot_actual_vs_predicted_by_equations
)

for api_number in unique_well_APIs_list:
      #Subset the dataframe by API Number
      production_time_series=df[df.API==api_number]
      #Get the highest value of production in the first 90 dayss of production, to use as qi value

      qi=get_max_initial_production(production_time_series, 300, 'filtered', 'drill_date')




      #Hyperbolic curve fit the data to get best fit equation
      popt_hyp, pcov_hyp=curve_fit(hyperbolic_equation, production_time_series['day'], 
                                   production_time_series['filtered'],bounds=(0, [qi,2,20]))
      print('Hyperbolic Fit Curve-fitted Variables: qi='+str(popt_hyp[0])+', b='+str(popt_hyp[1])+', di='+str(popt_hyp[2]))



      #Hyperbolic fit results
      production_time_series.loc[:,'Hyperbolic_Predicted']=hyperbolic_equation(production_time_series['day'], 
                                *popt_hyp)


      #Declare the x- and y- variables that we want to plot against each other
      y_variables=['filtered', "Hyperbolic_Predicted"]
      x_variable='day'
      #Create the plot title
      plot_title='filtered'+' Production for Well API '+str(api_number) 
      #Plot the data to visualize the equation fit
      plot_actual_vs_predicted_by_equations(production_time_series, x_variable, y_variables, plot_title)
