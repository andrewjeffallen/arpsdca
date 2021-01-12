def hyperbolic_equation(t, qi, b, di):
    return qi/((1.0+b*di*t)**(1.0/b))
    
    
def get_min_or_max_value_in_column_by_group(dataframe, group_by_column, calc_column, calc_type):
    value=dataframe.groupby(group_by_column)[calc_column].transform(calc_type)
    return value
    
def get_max_initial_production(df, number_first_days, variable_column, date_column):
    #sort dates
    df=df.sort_values(by=date_column)
    df_beginning_production=df.head(number_first_days)
    return df_beginning_production[variable_column].max()
    
def plot_actual_vs_predicted_by_equations(df, x_variable, y_variables, plot_title):
    """
    Plot Results of graphs
    """
    df.plot(x=x_variable, y=y_variables, title=plot_title)
    plt.show()
    
def remove_nan_and_zeroes_from_columns(df, variable):
    """
    remove nans and nulls
    """
    filtered_df = df[(df[variable].notnull()) & (df[variable]>0)]
    return filtered_df
