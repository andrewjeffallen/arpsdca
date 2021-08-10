import sklearn.metrics as metrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler

# Simple comparison of LR vs MLP on cleaned time series datasets


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


# Spot Check Algorithms


def compare_algorithms(x_train, y_train):
    models = []
    models.append(("LR", LinearRegression()))
    models.append(
        ("NN", MLPRegressor(solver="lbfgs", max_iter=10000))
    )  # neural network
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


def rmse(actual, predict):
    predict = np.array(predict)
    actual = np.array(actual)
    distance = predict - actual
    square_distance = distance ** 2
    mean_square_distance = square_distance.mean()
    score = np.sqrt(mean_square_distance)
    return score


def fit_model(model, X_test, y_test):

    if model == "MLP":
        mlp_model = MLPRegressor(solver="lbfgs", max_iter=10000)
        mlp_model.fit(X_test, y_test)

        mlp_y_true = y_test.values
        mlp_y_pred = mlp_model.predict(X_test)
        return regression_results(mlp_y_true, mlp_y_pred)

    elif model == "LR":
        lr_model = LinearRegression()
        lr_model.fit(X_test, y_test)

        lr_y_true = y_test.values
        lr_y_pred = lr_model.predict(X_test)
        return regression_results(lr_y_true, lr_y_pred)


def create_pca_df(data, dataframe, num_pc, time_horizon, unique_key, period):
    """
    data: well parameters pandas dataframe
    dataframe: time series pandas dataframe
    num_pc: the number of principal components
    time_horizon: int, time horizon to extract features from
    unique_key =
    """
    df_all["month_online"] = df_all.groupby(by=unique_key).cumcount() + 1
    df = df_all.groupby(by="API").head(time_horizon)
    df_wide = df.pivot(index="API", columns="month_online", values="Oil_MA")
    df_wide = df_wide.dropna()
    features = df_wide.columns[:num_pc]

    x = df_wide.loc[:, features].values

    x = StandardScaler().fit_transform(x)


    pc_cols = ", ".join([f"PC{i}" for i in range(1, num_pc + 1)])

    principalDf = pd.DataFrame(
        data=principalComponents, columns=[f"{pc_cols}"], index=df_wide.index
    )
    df_test = principalDf.merge(data, left_index=True, right_on="API")

    pcs = PCA()
    pcs.fit(preprocessing.scale(df_wide.loc[:, features].dropna(axis=0)))
    pcsSummary_df = pd.DataFrame(
        {
            "Standard deviation": np.sqrt(pcs.explained_variance_),
            "Proportion of variance": pcs.explained_variance_ratio_,
            "Cumulative proportion": np.cumsum(pcs.explained_variance_ratio_),
        }
    )
    pcsSummary_df = pcsSummary_df.transpose()
    pcsSummary_df.columns = [
        "PC{}".format(i) for i in range(1, len(pcsSummary_df.columns) + 1)
    ]
    pcsSummary_df.round(3)
