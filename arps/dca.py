import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from arps.utils import *
from lmfit import Model

# Use fitted equation from arps.fit_arps to create EUR prediction with CI intervals
class ArpsCurve:
    def predict_arps(
        prd_time_series,
        well_nm,
        liquid,
        qi_min,
        b_min,
        di_min,
        qi_max,
        b_max,
        di_max,
        sigma_fit,
        sigma_pred,
        pred_interval,
    ):
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
        df = pd.read_csv(f"{prd_time_series}.csv")

        df = df.astype({"api": str, f"{liquid}": float})
        df = df[(df[f"{liquid}"].notnull()) & (df[f"{liquid}"] > 0)]
        df["days"] = df.groupby("api").cumcount() + 1

        filtered_df = df[df.api == f"{well_nm}"]
        cumsum_days = filtered_df["days"]
        prod = filtered_df[f"{liquid}"]

        # plot data

        plt.plot(cumsum_days, prod, label=f"{liquid}", linewidth=1)

        # build Model
        hmodel = Model(hyperbolic_equation)

        # create lmfit Parameters, named from the arguments of `hyperbolic_equation`
        # note that you really must provide initial values.

        # params = hmodel.make_params(qi=431.0371968722894, b=0.5443981508109322, di=0.006643764565975722)

        params = hmodel.make_params(qi=qi_max, b=b_max, di=di_max)

        # set bounds on parameters
        params["qi"].min = qi_min
        params["b"].min = b_min
        params["di"].min = di_min

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
        print("R-Square: ", str(round(r2 * 100, 3)) + "%")

        # plot best fit: not that great of fit, really
        plt.plot(cumsum_days, result.best_fit, "r--", label="fit", linewidth=3)

        # calculate the (1 sigma) uncertainty in the predicted model
        # and plot that as a confidence band
        dprod = result.eval_uncertainty(result.params, sigma=sigma_fit)
        plt.fill_between(
            cumsum_days,
            result.best_fit - dprod,
            result.best_fit + dprod,
            color="#AB8888",
            label="uncertainty band of fit",
        )

        # now evaluate the model for other values, predicting future values
        future_days = np.array(np.arange(max(cumsum_days + 1), pred_interval))
        future_prod = result.eval(t=future_days)
        eur = sum(prod) + sum(future_prod)

        plt.plot(future_days, future_prod, "k--", label="prediction")

        # ...and calculate the 1-sigma uncertainty in the future prediction
        # for 95% confidence level, you'd want to use `sigma=2` here:
        future_dprod = result.eval_uncertainty(t=future_days, sigma=sigma_pred)

        # print("### Prediction\n# Day  Prod     Uncertainty")

        # for day, prod, eps in zip(future_days, future_prod, future_dprod):
        #     print(" {:.1f}   {:.1f} +/- {:.1f}".format(day, prod, eps))

        plt.fill_between(
            future_days,
            future_prod - future_dprod,
            future_prod + future_dprod,
            color="#ABABAB",
            label="uncertainty band of prediction",
        )

        plt.legend(loc="upper right")
        print("EUR: ", eur)
        plt.show()

    def execute_arps(prod_data, days, liquid, b_upper, di_bound, **kwargs):
        well_list = prod_data.api.unique()
        for api_number in well_list:
            prod_ts = prod_data[prod_data.api == api_number]
            prod_ts = prod_ts[(prod_ts[liquid].notnull()) & (prod_ts[liquid] > 0)]
            prod_ts["day"] = prod_ts.groupby(by="API").cumcount() + 1
            # convert API & Date to string and liquid to float
            prod_ts = prod_ts.astype({"API": str, "day": int, f"{liquid}": float})
            # Get the highest value of production in the first {days} days of production, to use as qi value
            qi = get_qi(prod_ts, days, liquid, "day")

            # Hyperbolic curve fit the data to get best fit equation
            popt_hyp, pcov_hyp = curve_fit(
                hyperbolic_equation,
                prod_ts["day"],
                prod_ts[liquid],
                bounds=(0, [qi, b_upper, di_bound]),
            )
            # print('Hyperbolic Fit Curve-fitted Variables: qi='+str(popt_hyp[0])+', b='+str(popt_hyp[1])+', di='+str(popt_hyp[2]))
            # Hyperbolic fit results
            prod_ts.loc[:, "hyperbolic_predicted"] = hyperbolic_equation(
                prod_ts["day"], *popt_hyp
            )
            y = prod_ts[liquid]
            y_fit = prod_ts["hyperbolic_predicted"]
            # residual sum of squares
            ss_res = np.sum((y - y_fit) ** 2)
            # total sum of squares
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            # r-squared
            r2 = 1 - (ss_res / ss_tot)
            print("Hyperbolic Fit Curve-fitted Variables:")
            print(
                "r2="
                + str(round(r2 * 100, 3))
                + "%"
                + " qi="
                + str(popt_hyp[0])
                + ", b="
                + str(popt_hyp[1])
                + ", di="
                + str(popt_hyp[2])
            )
            # Declare the x- and y- variables that we want to plot against each other
            y_variables = [f"{liquid}", "hyperbolic_predicted"]
            x_variable = "day"
            # Create the plot title
            plot_title = "liquid" + " Production for " + str(api_number)
            # Plot the data to visualize the equation fit
            plot_dca(prod_ts, x_variable, y_variables, plot_title)
