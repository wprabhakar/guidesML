import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# pre-processing
from sklearn.preprocessing import MinMaxScaler

# model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM # for CPU
#from keras.layers import CuDNNLSTM # for GPU

# evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt


# hyperparameters
WINDOW_SIZE = 2
REPEATS = 10
BATCH_SIZE = 1
N_EPOCH = 500
N_NEURONS = 2

###############################################################################
#                              HELPER FUNCTIONS                               #
###############################################################################

def plot_price_against_time(df):
    """
    Plots a graph of price vs. time (in quarters).

    Args:
        df (DataFrame) - combined dataset of original data and predicted data
    """
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ax.xaxis.set_major_locator(plt.MaxNLocator(20))    # to restrict the number of labels shown on the x-axis
    #ax.axhline(y=0, color='black', linewidth=1, linestyle='--')
          
    ax.plot(df.iloc[:116, :], color='blue')
    ax.plot(df.iloc[115:, :], color='red')
    ax.set(xlabel='Quarter', ylabel='Price per sqm', title='Price per sqm vs. quarter')

def timeseries_to_supervised(series, window_size):
    """
    Transforms a sequence into a supervised learning dataset
    
    Args:
        series (array) - the input sequence
        window_size (int) - the window sizes to create
    Returns:
        result (DataFrame) -  a supervised learning dataset, of window size 
        as features and the current value as the target
    """
    result = series.copy()
    for i in range(window_size):
        result = pd.concat([result, series.shift(-(i+1))], axis=1)

    result.dropna(inplace=True)
    return result

# inverts the scaling to get the original value
def invert_scale(scaler, X, value):
    """
    Inverts the scaling to get the original value.

    Args:
        scaler (MinMaxScaler) - scaler previously fit to training set
        X - 
        value - desired value to be inversely scaled
    Returns:
        inverted - original value after inverse scaling
    """
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# trains the LSTM model with the specified hyperparameters
def fit_lstm(train_sc, batch_size, nb_epoch, neurons):
    """
    Trains the LSTM model with the dataset and specified hyperparameters.
    
    Args:
        train_sc (array) - training set
        batch_size (int) - specified batch size
        nb_epoch (int) - number of epochs to run for model fitting
        neurons (int) - number of neurons in the LSTM network
    Returns:
        model (keras.Sequential) - trained LSTM network     
    """
    X = train_sc[:, :-1]
    y = train_sc[:, -1]
    X_rnn = X.reshape(X.shape[0], 1, X.shape[1]) # [samples, time steps, features]
    
    model = Sequential()
    # stateful=True ensures state is maintained; used when treating consecutive batches as consecutive inputs
    model.add(LSTM(neurons,
                  batch_input_shape=(batch_size, X_rnn.shape[1], X_rnn.shape[2]),
                  stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    # fit model
    # reset_states() is manually called after every epoch as the network is stateful; 
    # helps to make consecutive model calls independent
    for i in range(nb_epoch): 
        model.fit(X_rnn, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()

    return model

def predict_lstm(model, batch_size, X_test_sc):
    """
    Given a trained model and one feature row, output a prediction.

    Args:
        model (keras.Sequential) - trained LSTM network
        batch_size - specified batch size
        X_test_sc - 1 feature row
    Returns:
        yhat (float) - predicted value
    """

    # reset_states() is not called as the model should build up state 
    # following each prediction
    X_test_rnn = X_test_sc.reshape(1, 1, len(X_test_sc))
    yhat = model.predict(X_test_rnn, batch_size=batch_size)
    return yhat[0,0]

###############################################################################
#                            SEQUENCE FUNCTIONS                               #
###############################################################################

def transform_dataset(df_ts):
    """
    Transforms a dataset to stationary data (using 1st order difference), 
    and modifies the sequence to a supervised learning dataset.

    Args:
        df_ts (DataFrame)  - dataset in time-series format
    Returns:
        df_sup (DataFrame) - stationary data in a supervised learning dataset
    """
    
    # use diff() to transform the data to stationary
    df_sup = timeseries_to_supervised(pd.DataFrame(df_ts.diff().values),
                                            WINDOW_SIZE)
    return df_sup

def train_test_split(df_sup, split_proportion):
    """
    Splits the dataset into train and test sets based on the specified train_size

    Args:
        df_sup (DataFrame) - dataset
        train_size (float) - train/test split proportion
    Returns:
        train (DataFrame) - training set
        test (DataFrame) - test set
    """
    train_size = int(len(df_sup) * split_proportion)
    train, test = df_sup[0:train_size], df_sup[train_size:len(df_sup)]
    return train, test

def scale(train, test):
    """
    Scales the train and test sets using MinMaxScaler to restrict the features
    between -1 and 1.
    Test data is scaled using the fit of the scaler on the training data, 
    so as to ensure the min/max values of the test data do not influence the model.

    Args:
        train (DataFrame) - training set
        test (DataFrame) - test set
    Returns:
        scaler (MinMaxScaler) - scaler fit to training set
        train_sc (array) - scaled training set
        test_sc (array) - scaled test set
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(train)
    train_sc = scaler.transform(train)
    test_sc = scaler.transform(test)
    return scaler, train_sc, test_sc

def model_fit_and_predict(repeats, df_sup, test, train_sc, test_sc, scaler, batch_size, n_epoch, n_neurons):
    """
    Contains the meat of the process.
    Runs the fit-and-predict sequence multiple times (as neural networks are 
    greatly affected by the random initial conditions) and store the performance
    of the model during each run.

    Args:
        repeats (int) - number of times to run the sequence
        df_sup (DataFrame) - dataset
        test (array) - test set
        train_sc (array) - scaled training set
        test_sc (array) - scaled test set
        scaler (MinMaxScaler) - scaler fit to training set
        batch_size (int) - specified batch size
        n_epoch (int) - number of epochs to run for model fitting
        n_neurons (int) - number of neurons in the LSTM network
    Returns:
        error_scores (list) - list of error scores on the model for each run
        predictions_full (list) - list of predictions made for each run
        models_full (list) - list of models trained for each run
    """
    # variables to store outputs
    error_scores = list()
    predictions_full = list()
    models_full = list()

    for i in range(repeats):
        model = fit_lstm(train_sc, batch_size, n_epoch, n_neurons)
        predictions = list()
        
        # walk-forward model validation
        # (use model to make a forecast for this time step, but use the actual 
        # value from the test set to make the forecast for the next time step 
        # instead of using the previous forecast)
        for j in range(len(test_sc)):
            X_test_sc = test_sc[j, :-1]
            y_test_sc = test_sc[j, -1]
            yhat = predict_lstm(model, batch_size, X_test_sc)
            
            yhat = invert_scale(scaler, X_test_sc, yhat)
            
            predictions.append(yhat)

        # error on the 1st order diff
        rmse = sqrt(mean_squared_error(test.iloc[:, -1], predictions))
        #print('%d) Test RMSE: %.3f' % (i+1, rmse))
        error_scores.append(rmse)
        
        predictions_full.append(predictions)
        models_full.append(model)
        
    return error_scores, predictions_full, models_full

def get_best_run_info(error_scores, predictions, models):
    """
    Find the best model and the associated predictions based on the lowest
    RMSE score.
    
    Args:
        error_scores (list) - list of error scores on the model for each run
        predictions (list) - list of predictions made for each run
        models (list) - list of models trained for each run
    Returns:
        model_best (keras.Sequential) - best model
        prediction_best (array) - corresponding predictions made by the best model
    """
    best_run_index = np.argmin(error_scores)
    model_best = models[best_run_index]
    prediction_best = predictions[best_run_index]
    return model_best, prediction_best

def compare_against_test_set(df_ts, test, prediction_best):
    """
    Compare the predicted values against the actual values in the dataset.

    Args:
        df_ts (DataFrame) - original dataset
        test (DataFrame) - test set, after train-test split
        prediction_best (array) - predicted values on the test set
    Returns:
        df_reconstructed (DataFrame) - contains a 'truth' and 'prediction' column
    """
    y_test = test.iloc[:, -1].values
    y_pred = prediction_best

    test_start_index = test.index[0] # start of test data
    test_end_index = test.index[-1] # end of test data
    train_end_index = test_start_index-1 # end of training data
    # last value of training data
    test_start_value = df_ts.iloc[train_end_index]['avg_price_per_sqm']

    # compute the predictions using cumulative sum
    df_reconstructed = pd.DataFrame({'predictions': [test_start_value] + y_pred}).cumsum()
    df_reconstructed['truth'] = df_ts.iloc[train_end_index:test_end_index+1].values
    # predictions for 1st order difference
    #df_diff_reconstructed = pd.DataFrame({'predictions': y_pred})
    #df_diff_reconstructed['truth'] = y_test

    return df_reconstructed


###############################################################################
#                            MAIN FUNCTIONS                                   #
###############################################################################

def train(df_ts):
    """
    Main method.
    Performs the following:
        1. Pre-processing
            > Transform to supervised learning
            > Split into train and test sets
            > Scale the data to the range (-1, 1)
        2. Fit and train model
            > LSTM network
        3. Evaluate model

    Args:
        df_ts (DataFrame) - dataset in time-series format
    Returns:
        scaler (MinMaxScaler) - scaler fit to training set
        model_best (keras.Sequential) - selected best model
        df_reconstructed (DataFrame) - contains actual and predicted values
        last_value (1x2 array) - final value of dataset, to facilitate 
                                 future forecasts
    """

    # pre-processing
    df_sup = transform_dataset(df_ts)
    train, test = train_test_split(df_sup, 0.75)
    scaler, train_sc, test_sc = scale(train, test)

    # model
    error_scores, predictions, models = model_fit_and_predict(REPEATS,
                                                              df_sup,
                                                              test,
                                                               train_sc, 
                                                              test_sc,
                                                              scaler,
                                                              BATCH_SIZE, 
                                                              N_EPOCH, 
                                                              N_NEURONS)
    model_best, prediction_best = get_best_run_info(error_scores, 
                                                    predictions, 
                                                    models)

    # evaluate
    df_reconstructed = compare_against_test_set(df_ts, test, prediction_best)

    # final value of dataset (last 2 columns in the last row)
    last_value = test_sc[len(test_sc)-1, 1:]

    return scaler, model_best, df_reconstructed, last_value
    
def forecast(scaler, model, last_value):
    """
    Makes forecasts in the future using the trained model.

    Args:
        n_forecasts (int) - number of forecasts to make
        scaler (MinMaxScaler) - scaler fit to training set
        model (keras.Sequential) - trained LSTM network
        last_value (1x2 array) - last value (scaled) in the original dataset
    Returns:
        pred_future (array) - forecasted values in the future
    """

    pred_sc = list() # nx3-array
    pred_sc.append(last_value)
    pred_future = list() # nx1-array

    # predict for next quarters
    for i in range(20): 
        yhat = predict_lstm(model, 1, pred_sc[i])
        
        # invert the scaling of the prediction and store separately
        yhat_inv = invert_scale(scaler, pred_sc[i], yhat)
        pred_future.append(yhat_inv)
        
        # populate the scaled table to facilitate further predictions
        pred_sc[i] = np.append(pred_sc[i], yhat)
        pred_sc.append(pred_sc[i][len(pred_sc[i])-2:len(pred_sc[i])])

    return pred_future

def plot_present_with_forecast(df_ts, pred_future):
    """
    Appends the future forecasts to the original dataset and plots the
    graph of the combined dataset.
    
    Args:
        df_ts (DataFrame) - original dataset
        pred_future (array) - predicted future values
    Returns:
        avg_pred_price_pct_change (float) - average percentage predicted change
            of price in the prediction timeframe
    """
    qtr_future = ['2019Q1', '2019Q2', '2019Q3', '2019Q4',
                  '2020Q1', '2020Q2', '2020Q3', '2020Q4',
                  '2021Q1', '2021Q2', '2021Q3', '2021Q4',
                  '2022Q1', '2022Q2', '2022Q3', '2022Q4',
                  '2023Q1', '2023Q2', '2023Q3', '2023Q4',
                 ]

    df_future = pd.DataFrame({'avg_price_per_sqm': 
                    [df_ts.tail(1).values[0][0]] + pred_future}).cumsum()

    # remove the 1st element, which is the last element of the original dataset
    df_future.drop(0, inplace=True)
    # set index to be the quarter, in accordance with the original df
    df_future['quarter'] = qtr_future
    df_future.set_index('quarter', drop=True, inplace=True)

    # combine the original and predicted values
    df_comb = pd.concat([
        df_ts,
        df_future
    ])

    plot_price_against_time(df_comb)

    # get the mean of the price change
    pred_price_pct_change = df_future['avg_price_per_sqm'].pct_change()
    avg_pred_price_pct_change = pred_price_pct_change.mean()

    return avg_pred_price_pct_change