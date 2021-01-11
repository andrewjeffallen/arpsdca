def hyperbolic_equation(t, qi, b, di):
    """
    Hyperbolic decline curve equation
    Arguments:
        t: Float. Time since the well first came online, can be in various units 
        (days, months, etc) so long as they are consistent.
        qi: Float. Initial production rate when well first came online.
        b: Float. Hyperbolic decline constant
        di: Float. Nominal decline rate at time t=0
    Output: 
        Returns q, or the expected production rate at time t. Float.
    """
    return qi/((1.0+b*di*t)**(1.0/b))
    
    
  def get_min_or_max_value_in_column_by_group(dataframe, group_by_column, calc_column, calc_type):
    """
    This function obtains the min or max value for a column, with a group by applied. For example,
    it could return the earliest (min) RecordDate for each API number in a dataframe 
    Arguments:
        dataframe: Pandas dataframe 
        group_by_column: string. Name of column that we want to apply a group by to
        calc_column: string. Name of the column that we want to get the aggregated max or min for
        calc_type: string; can be either 'min' or 'max'. Defined if we want to pull the min value 
        or the max value for the aggregated column
    Outputs:
        value: Depends on the calc_column type.
    """
    value=dataframe.groupby(group_by_column)[calc_column].transform(calc_type)
    return value
    
  def get_max_initial_production(df, number_first_days, variable_column, date_column):
    """
   
    Arguments:
        df: our production data
        number_first_months: float. Number of months from time well began production
        get the max initial production rate qi 
        
        variable_column: String. Column name of either 'gas' or 'oil' that we want max initial value
        
        date_column: String. Column name for the date that the data was taken at: our name is drill_date
        
       
    """
    #sort dates
    df=df.sort_values(by=date_column)
    
    # x is our number of 90 to observe
    df_beginning_production=df.head(number_first_days)
    
    
    # return max value
    # create new datafram with values sorted
    return df_beginning_production[variable_column].max()
    
  def plot_actual_vs_predicted_by_equations(df, x_variable, y_variables, plot_title):
    """
Plot Results of graphs
    """
    df.plot(x=x_variable, y=y_variables, title=plot_title)
    plt.show()
    
    
