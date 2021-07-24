import numpy
import pandas 
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scikit-learn

from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler


def arps_neural_network(df):
        """
        df = production time series dataframe with 2 columns:
                1. drill_date (datetime object)
                2. oil_bbl (float)

        """
        dataframe = production_time_series=df[df.API==api_number]
        dataframe = dataframe[['oil_bbl']]

        for api_number in unique_well_APIs_list:
                #Subset the dataframe by API Number
                production_time_series=df[df.API==api_number]
                #Get the highest value of production in the first 3 months of production, to use as qi value
                dataframe = production_time_series=df[df.API==api_number]
                dataframe = dataframe[['oil_bbl']]
                qi=get_max_initial_production(production_time_series, 40, 'oil_bbl', 'drill_date')

                #Hyperbolic curve fit the data to get best fit equation
                popt_hyp, pcov_hyp=curve_fit(hyperbolic_equation, production_time_series['day'], 
                                             production_time_series['oil_bbl'],bounds=(0, [qi,2,20]))
                print('Hyperbolic Fit Curve-fitted Variables: qi='+str(popt_hyp[0])+', b='+str(popt_hyp[1]

                numpy.random.seed(3)
        # load the dataset
                dataset = dataframe.values
                dataset = dataset.astype('float32')
        # normalize the dataset
                scaler = MinMaxScaler(feature_range=(0, 1))
                dataset = scaler.fit_transform(dataset)
        # split into train and test sets
                train_size = int(len(dataset) * 0.6)
                test_size = len(dataset) - train_size
                train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
        # reshape into X=t and Y=t+1
                look_back = 3
                trainX, trainY = create_dataset(train, look_back)
                testX, testY = create_dataset(test, look_back)
        # reshape input to be [samples, time steps, features]
                trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
                testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
        # create and fit the LSTM network
                model = Sequential()
                model.add(LSTM(10, input_shape=(1, look_back)))
                model.add(Dense(1))
                model.compile(loss='mean_squared_error', optimizer='adam')
                model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
        # make predictions
                trainPredict = model.predict(trainX)
                testPredict = model.predict(testX)
        # invert predictions
                trainPredict = scaler.inverse_transform(trainPredict)
                trainY = scaler.inverse_transform([trainY])
                testPredict = scaler.inverse_transform(testPredict)
                testY = scaler.inverse_transform([testY])
        # calculate root mean squared error
                trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
                print('Train Score: %.2f RMSE' % (trainScore))
                testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
                print('Test Score: %.2f RMSE' % (testScore))
        # shift train predictions for plotting
                trainPredictPlot = numpy.empty_like(dataset)
                trainPredictPlot[:, :] = numpy.nan
                trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
        # shift test predictions for plotting
                testPredictPlot = numpy.empty_like(dataset)
                testPredictPlot[:, :] = numpy.nan
                testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
        # plot baseline and predictions
                plt.plot(scaler.inverse_transform(dataset))
                plt.plot(trainPredictPlot)
                plt.plot(testPredictPlot)
                plt.show()
