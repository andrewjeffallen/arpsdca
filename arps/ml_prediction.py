# Simple comparison of LR vs MLP on cleaned time series datasets

import sklearn.metrics as metrics
from sklearn.model_selection import TimeSeriesSplit


def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mse = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)
    print("explained_variance: ", round(explained_variance, 4))
    print("mean_squared_log_error: ", round(mean_squared_log_error, 4))
    print("r2: ", round(r2, 4))
    print("MAE: ", round(mean_absolute_error, 4))
    print("MSE: ", round(mse, 4))
    print("RMSE: ", round(np.sqrt(mse), 4))


time_series = pd.read_csv("daily_production.csv")
time_series[time_series.API == 42493326380000]
time_series = time_series.astype({"API": str, "OIL (BBL)": float})
time_series = time_series[
    (time_series["OIL (BBL)"].notnull()) & (time_series["OIL (BBL)"] > 0)
]

data = time_series[["D_DATE", "OIL (BBL)"]]

data["D_DATE"] = pd.to_datetime(data["D_DATE"])
data = data.set_index("D_DATE")


data_consumption = data[["OIL (BBL)"]]
# inserting new column with yesterday's consumption values
data_consumption.loc[:, "Yesterday"] = data_consumption.loc[:, "OIL (BBL)"].shift()
# inserting another column with difference between yesterday and day before yesterday's consumption values.
data_consumption.loc[:, "Yesterday_Diff"] = data_consumption.loc[:, "Yesterday"].diff()
# dropping NAs
data_consumption = data_consumption.dropna()


X_train = data_consumption[data_consumption.index < "2017"].drop(["OIL (BBL)"], axis=1)
y_train = data_consumption[data_consumption.index < "2017"]["OIL (BBL)"]


X_test = data_consumption[data_consumption.index >= "2017"].drop(["OIL (BBL)"], axis=1)
y_test = data_consumption[data_consumption.index >= "2017"]["OIL (BBL)"]

# Spot Check Algorithms
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score

models = []
models.append(("LR", LinearRegression()))
models.append(("NN", MLPRegressor(solver="lbfgs", max_iter=10000)))  # neural network
# Evaluate each model in turn
results = []
names = []
for name, model in models:
    tscv = TimeSeriesSplit(n_splits=10)
    cv_results = cross_val_score(model, X_train, y_train, cv=tscv, scoring="r2")
    results.append(cv_results)
    names.append(name)
    print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

# Compare Algorithms
plt.boxplot(results, labels=names)
plt.title("Algorithm Comparison")
plt.show()

from sklearn.metrics import make_scorer


def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score


mlp_model = MLPRegressor(solver="lbfgs", max_iter=10000)
mlp_model.fit(X_test, y_test)

mlp_y_true = y_test.values
mlp_y_pred = mlp_model.predict(X_test)
regression_results(mlp_y_true, mlp_y_pred)


lr_model = LinearRegression()
lr_model.fit(X_test, y_test)

lr_y_true = y_test.values
lr_y_pred = lr_model.predict(X_test)
regression_results(lr_y_true, lr_y_pred)
