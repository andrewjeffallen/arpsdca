# arpsdca

## Description
The `arpsdca` provides a python based solution for Arps decline-curve analysis on oil and gas data

At the core, this repository includes:
* fitting Arps equation
$$q(t) =  \frac{qi} {(1+b d_{i} t)^{1-b}}$$
via [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
* EUR calculation with abilities to bound variables
* automated graphical plotting via [matplotlib](https://matplotlib.org/tutorials/introductory/sample_plots.html#sphx-glr-tutorials-introductory-sample-plots-py)

## Usage

The [execute_arps](https://github.com/andrewjeffallen/arpsdca/blob/main/arps/fit_arps.py) allows you to find the `qi`, `b`, and `di` variables in Arps equation in an automated fashion using functions defined in the [arps](https://github.com/andrewjeffallen/arpsdca/blob/main/arps/dca.py) module

#### Using `arpsdca` on your production data

your dataset should have the general structure:

| API             | production_rate             |  drill_date   |
| :---            |    :---:                    |          ---: |
| Unique Well num |  rate in BBL, Gal, or Mcf   |   timestamp   |

You may have 1 or many wells in the dataset, the `execute_arps` will work regardless!


#### Using LSTM for production data

Using the same data structure as defined above, the [lstm.py](https://github.com/andrewjeffallen/arpsdca/blob/main/arps/lstm.py) module allows you to train, validate, test long short term memory recurrent neural network on your production data. 


## *Features coming soon*
#### Arps
* Bounding of the `di` variable in EUR calculation
* What-if scenario with corresponding side-by-side DCA plots and EUR values
* Microsoft Excel plugin and data export
#### LSTM
* Parameterized PCA as input neurons
* custom `cost_function` object
* improved speed and performance for training multiple time series 




