import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from arps.utils import *
from lmfit import Model

# Use fitted equation from arps.fit_arps to create EUR prediction with CI intervals
class ArpsModel:
    def __init__(
        self,
        qi,
        qi_min,
        di_nom,
        di_nom_min,
        di_nom_max,
        b,
        b_min,
        b_max,
        ts,
        production,
        resource_type,
    ):

        self.qi = qi
        self.qi_min = qi_min

        self.di_nom = di_nom
        self.di_nom_min = di_nom_min
        self.di_nom_max = di_nom_max

        self.b = b
        self.b_min = b_min
        self.b_max = b_max

        self.ts = ts
        self.production = production

        self.resource_type = resource_type

        if self.qi < 0.0:
            raise ValueError("Cannot have negative qi value")
        if self.di_nom < 0.0:
            raise ValueError("Cannot have negative Di Nominal Value")
        if self.b < 0.0 or self.b > b_threshold:
            raise ValueError(f"Invalid b: {self.b}, Cannot be aboe {self.b_threshold}")

        if self.ts <= 0:
            raise ValueError("Time series must contain 1 or more values")

    def forecast(self, sigma_fit, sigma_pred, periods):
        """
        Forecast using Arps Decline Model

        Args:
            sigma_fit (float):  Confidence level in fitted model
            sigma_pred (float):  Confidence level in predicted model
            periods (int): n periods to forecast
        """
        hmodel = Model(DeclineCurve.hyperbolic_equation)

        params = hmodel.make_params(qi=self.qi, b=self.b, di=self.di_nom)

        # set bounds on parameters
        params["qi"].min = self.qi_min
        params["b"].min = self.b_min
        params["di"].min = self.di_min

        result = hmodel.fit(self.production, params, t=self.ts)

        y = self.ts
        y_fit = result.best_fit

        ss_res = np.sum((y - y_fit) ** 2)
        # total sum of squares
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        # r-squared
        r2 = 1 - (ss_res / ss_tot)
        print(result.fit_report())
        print("R-Square: ", str(round(r2 * 100, 3)) + "%")

        self.dprod = result.eval_uncertainty(result.params, sigma=sigma_fit)
        self.future_days = np.array(np.arange(max(self.ts + 1), periods))
        self.production_forecast = result.eval(t=self.future_days)

        # Calculate the 1-sigma uncertainty in the future prediction
        # for 95% confidence level, you'd want to use `sigma=2` here:
        self.future_dprod = result.eval_uncertainty(
            t=self.future_days, sigma=sigma_pred
        )
        self.eur = sum(self.production) + sum(self.production_forecast)
        self.y_pred = y_fit

        print(
            {
                "Fitted Model Uncertainty": self.dprod,
                "Forecast Horizon (days)": self.future_days,
                "Forecast EUR": self.eur,
            }
        )
        return (
            self.dprod,
            self.future_days,
            self.production_forecast,
            self.future_dprod,
            self.eur,
            self.y_pred,
        )

    def make_plots(self):

        fp = self.forecast(self.sigma_fit, self.sigma_pred, self.periods)

        # Plot historical production
        plt.plot(self.ts, self.production, label=f"{self.resource_type}", linewidth=1)

        # plot fitted model
        plt.plot(self.ts, fp.y_pred, "r--", label="fit", linewidth=3)

        # Plot Uncertainty band of fit
        plt.fill_between(
            self.ts,
            self.y_pred - self.dprod,
            self.y_pred + self.dprod,
            color="#AB8888",
            label="uncertainty band of fit",
        )

        # Plot Predicted model
        plt.plot(fp.future_days, fp.production_forecast, "k--", label="prediction")

        plt.fill_between(
            fp.future_days,
            fp.production_forecast - fp.future_dprod,
            fp.production_forecast + fp.future_dprod,
            color="#ABABAB",
            label="uncertainty band of prediction",
        )

        plt.legend(loc="upper right")
        print("EUR: ", fp.eur)
        plt.show()

    def hyperbolic_equation(self):
        return self.qi / (
            (1.0 + self.b * self.di_nom * self.ts) ** (1.0 / max(self.b, 1.0e-50))
        )

    def plot_dca(df, x_variable, y_variables, plot_title):
        """
        Plot Results of graphs
        """
        plt.style.use("fivethirtyeight")
        mpl.rcParams["lines.linewidth"] = 2
        df.plot(x=x_variable, y=y_variables, title=plot_title)
        plt.show()

    def plot_dca(df, x_variable, y_variables, plot_title):
        """
        Plot Results of graphs
        """
        plt.style.use("fivethirtyeight")
        mpl.rcParams["lines.linewidth"] = 2
        df.plot(x=x_variable, y=y_variables, title=plot_title)
        plt.show()

    def fit_arps(self):

        # get qi_max
        qi = self.qi

        # Hyperbolic curve fit the data to get best fit equation
        popt_hyp, pcov_hyp = curve_fit(
            self.hyperbolic_equation,
            self.ts,
            self.production,
            bounds=([0, self.b_min, self.di_nom_min], [qi, self.b_max, self.di_nom_max]),
        )
        # print('Hyperbolic Fit Curve-fitted Variables: qi='+str(popt_hyp[0])+', b='+str(popt_hyp[1])+', di='+str(popt_hyp[2]))
        # Hyperbolic fit results
        hyperbolic_predicted = self.hyperbolic_equation(
            self.ts, *popt_hyp
        )
        y = self.production
        y_fit = hyperbolic_predicted
        
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
        y_variables = [self.production, hyperbolic_predicted]
        x_variable = self.ts
        plot_title = f""" Production Plots
        qi={str(popt_hyp[0])}, b={str(popt_hyp[1])}, di={str(popt_hyp[2])}
        """
        df = pd.DataFrame(
            columns = ['ts', 'y_fit', 'y' ],
            rows = [self.ts, y_fit, self.production]
            )
        # Plot the data to visualize the equation fit
        self.plot_dca(df, x_variable, y_variables, plot_title)
