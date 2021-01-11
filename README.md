# arpsdca

## Description
The `arpsdca` provides a python based solution for Arps decline-curve analysis on oil and gas data

At the core, this repository includes:
* fitting Arps equation via [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html)
* EUR calculation with abilities to bound variables
* automated graphical plotting via [matplotlib](https://matplotlib.org/tutorials/introductory/sample_plots.html#sphx-glr-tutorials-introductory-sample-plots-py)

## Usage

The [execute_arps](https://github.com/andrewjeffallen/arpsdca/blob/main/arps/fit_arps.py) allows you to find the `qi`, `b`, and `di` variables in Arps equation in an automated fashion.

#### Using `arpsdca` on your production data

| API          | production_rate  | Right-aligned |
| :---         |     :---:        |          ---: |
| git status   | git status       | git status    |
| git diff     | git diff         | git diff      |
