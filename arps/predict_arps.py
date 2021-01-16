import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt

from lmfit import Model
from arps.dca_utils import hyperbolic_equation

# Use fitted equation from arps.fit_arps to create EUR prediction with CI intervals

def predict_arps(prd_time_series, well_nm, liquid, qi_min, b_min, di_min,qi_max, b_max, di_max,sigma_fit,sigma_pred,pred_interval):
  """
  prd_time_series: (str) a production time series in form of a csv
  well_nm: (str, int) the API, or well ID/ name 
  liquid: (str) the type of liquid to calculuate EUR
  qi_min: (int) qi lower bound 
  b_min: (int) bi lower bound 
  di_min: (int) di lower bound 
  qi_max: (int) qi upper bound  
  b_max: (int) bi upper bound 
  di_max: (int) di upper bound 
  sigma_fit: (int) significance level for fitted curve
  sigma_pred: (int) significance level for predicted curve
  pred_interval: (float) future number of days to predict 
  
  """
  df = pd.read_csv(f'{prd_time_series}.csv')
  
  df = df.astype({"API": str,f'{liquid}':float})
  df = df[(df[f'{liquid}'].notnull()) & (df[f'{liquid}']>0)]
  df['days'] = df.groupby('API').cumcount()+1
  
  filtered_df = df[df.API==f'{well_nm}']
  cumsum_days = filtered_df['days']
  prod = filtered_df[f'{liquid}']

  # plot data
  plt.plot(cumsum_days, prod,label='data',linewidth=1)

  # build Model
  hmodel = Model(hyperbolic_equation)

  # create lmfit Parameters, named from the arguments of `hyperbolic_equation`
  # note that you really must provide initial values.

  #params = hmodel.make_params(qi=431.0371968722894, b=0.5443981508109322, di=0.006643764565975722)

  params = hmodel.make_params(qi=qi_max, b=b_max, di=di_max)



  # set bounds on parameters
  params['qi'].min=qi_min
  params['b'].min=b_min
  params['di'].min=di_min

  # do fit, print resulting parameters
  result = hmodel.fit(prod, params, t=cumsum_days)

  y = prod
  y_fit = result.best_fit

  ss_res = np.sum((y - y_fit) ** 2)
  # total sum of squares
  ss_tot = np.sum((y - np.mean(y)) ** 2)
  # r-squared
  r2 = 1 - (ss_res / ss_tot)
  print(result.fit_report())
  print("R-Square: ",r2)

  # plot best fit: not that great of fit, really
  plt.plot(cumsum_days, result.best_fit, 'r--', label='fit',linewidth=3)

  # calculate the (1 sigma) uncertainty in the predicted model
  # and plot that as a confidence band
  dprod = result.eval_uncertainty(result.params, sigma=sigma_fit)   
  plt.fill_between(cumsum_days,
                   result.best_fit-dprod,
                   result.best_fit+dprod,
                   color="#AB8888",
                   label='uncertainty band of fit')

  # now evaluate the model for other values, predicting future values
  future_days = np.array(np.arange(max(cumsum_days+1),pred_interval))
  future_prod = result.eval(t=future_days)
  eur = sum(prod)+sum(future_prod)

  plt.plot(future_days, future_prod, 'k--', label='prediction')

  # ...and calculate the 1-sigma uncertainty in the future prediction
  # for 95% confidence level, you'd want to use `sigma=2` here:
  future_dprod = result.eval_uncertainty(t=future_days, sigma=sigma_pred)

  #print("### Prediction\n# Day  Prod     Uncertainty")

  # for day, prod, eps in zip(future_days, future_prod, future_dprod):
  #     print(" {:.1f}   {:.1f} +/- {:.1f}".format(day, prod, eps))

  plt.fill_between(future_days,
                   future_prod-future_dprod,
                   future_prod+future_dprod,
                   color="#ABABAB",
                   label='uncertainty band of prediction')

  plt.legend(loc='upper right')
  print("EUR: ", eur)
  plt.show()
  
if __name__ == "__main__":
  
  prd_time_series = sys.argv[1]
  well_nm = sys.argv[2]
  liquid = sys.argv[3]
  qi_min = sys.argv[4]
  b_min = sys.argv[6]
  di_min = sys.argv[7]
  qi_max = sys.argv[8]
  b_max = sys.argv[9]
  di_max = sys.argv[10]
  sigma_fit = sys.argv[11]
  sigma_pred = sys.argv[12]
  pred_interval = sys.argv[13]
      
  return_code = predict_arps(prd_time_series, well_nm, liquid, qi_min, b_min, di_min,qi_max, b_max, di_max,sigma_fit,sigma_pred,pred_interval)
  
  if return_code == 0:
    sys.exit(0)
  
  else:
    raise SystemError(f"Error {return_code}")
