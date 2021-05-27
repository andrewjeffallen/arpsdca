import matplotlib.pyplot as plt
import matplotlib as mpl


class DeclineCurve(self):

 def __init__(self,hyperbolic_equation,get_qi,plot_dca):
  self.hyperbolic_equation=hyperbolic_equation
  self.get_qi=get_qi
  self.plot_dca=plot_dca
  
  def hyperbolic_equation(t, qi, b, di):
        return qi/((1.0+b*di*t)**(1.0/max(b, 1.e-50)))
        
  def get_qi(df, days, var_col, date_col):
      df=df.sort_values(by=var_col)
      df_init_prod=df.head(days)
      return df_init_prod[date_col].max()
        
  def plot_dca(df, x_variable, y_variables, plot_title):
      """
      Plot Results of graphs
      """
      plt.style.use('fivethirtyeight')
      mpl.rcParams['lines.linewidth'] = 2
      df.plot(x=x_variable, y=y_variables, title=plot_title)
      plt.show()
