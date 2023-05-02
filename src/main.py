# Import necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf
from numpy import mat
from sklearn.ensemble import RandomForestRegressor, VotingClassifier, GradientBoostingRegressor, VotingRegressor
from finta import TA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
import xgboost as xgb

# Define the ticker symbols and period of interest
tickers = ["AAPL", "MSFT", "AMZN"]
period = "1y"
# Create an empty DataFrame with column names
df_all_predicts = pd.DataFrame(columns=['Stock Name', 'Actual', 'Prediction'])
df_all_companies = pd.DataFrame(columns = ['Stock', 'Open', 'High', 'Low', 'Close', 'Volume'])

for ticker in tickers:
    # Fetch the data using yfinance
    data = yf.download(ticker, period=period)

    # Remove the current month's data
    data = data[data.index.month != pd.Timestamp.now().month]

    # Save the data as a CSV file
    # data.to_csv(f"{ticker}_{period}.csv", index=True)


    df = pd.DataFrame(
        {'Stock': ticker, 'Open': data['Open'], 'High': data['High'], 'Low': data['Low'], 'Close': data['Close'],
         'Volume': data['Volume']})


    df_all_companies = df_all_companies._append(df)

    # List of symbols for technical indicators
    INDICATORS = ['RSI', 'MACD', 'STOCH', 'ADL', 'ATR', 'MOM', 'MFI', 'ROC', 'OBV', 'CCI', 'EMV', 'VORTEX']

    """
    Next we pull the historical data using yfinance
    Rename the column names because finta uses the lowercase names
    """
    data.rename(columns={"Close": 'close', "High": 'high', "Low": 'low', 'Volume': 'volume', 'Open': 'open'},
                inplace=True)


    def _get_indicator_data(data):
        """
        Function that uses the finta API to calculate technical indicators used as the features
        :return:
        """

        for indicator in INDICATORS:
            ind_data = eval('TA.' + indicator + '(data)')
            if not isinstance(ind_data, pd.DataFrame):
                ind_data = ind_data.to_frame()
            data = data.merge(ind_data, left_index=True, right_index=True)
        data.rename(columns={"14 period EMV.": '14 period EMV'}, inplace=True)

        # Also calculate moving averages for features
        data['ema50'] = data['close'] / data['close'].ewm(50).mean()
        data['ema21'] = data['close'] / data['close'].ewm(21).mean()
        data['ema15'] = data['close'] / data['close'].ewm(14).mean()
        data['ema5'] = data['close'] / data['close'].ewm(5).mean()

        # Instead of using the actual volume value (which changes over time), we normalize it with a moving volume average
        data['normVol'] = data['volume'] / data['volume'].ewm(5).mean()

        # Remove columns that won't be used as features
        del (data['open'])
        del (data['high'])
        del (data['low'])
        del (data['volume'])
        del (data['Adj Close'])

        return data


    data = _get_indicator_data(data)


    def _train_xgboost(X_train, y_train, X_test, y_test):
        # Define model
        xgb_model = xgb.XGBRegressor()

        # Define hyperparameters to tune
        param_grid = {
            'n_estimators': [100, 500, 1000],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.5],
        }

        # Perform grid search
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        # Save best model
        xgb_best = grid_search.best_estimator_
        # Make predictions on the testing data
        y_pred = xgb_best.predict(X_test)
        return xgb_best


    def _train_random_forest(X_train, y_train, X_test, y_test):
        """
        Function that uses random forest classifier to train the model
        :return:
        """

        # Define parameter grid for GridSearchCV
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        # Create Random Forest Regressor
        rf = RandomForestRegressor()

        # Use GridSearchCV to find best hyperparameters
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Save best model
        rf_best = grid_search.best_estimator_
        # Make predictions on the testing data
        y_pred = rf_best.predict(X_test)

        return rf_best


    def _train_svr(X_train, y_train, X_test, y_test):
        svr = SVR(kernel='rbf')

        # Define the parameter grid for Grid Search
        param_grid = {'C': [0.1, 1, 10, 100],
                      'gamma': [1, 0.1, 0.01, 0.001],
                      'epsilon': [0.1, 0.01, 0.001, 0.0001]}

        # Perform Grid Search to find the best hyperparameters
        grid_search = GridSearchCV(svr, param_grid, cv=5)
        grid_search.fit(X_train, y_train)

        # Get the best hyperparameters from Grid Search
        best_params = grid_search.best_estimator_

        # Make predictions on the testing set
        predictions = best_params.predict(X_test)

        return best_params


    def _ensemble_model(rf_model, knn_model, xgb_model, X_train, y_train, X_test, y_test):
        # Create a dictionary of our models
        estimators = [('svr', knn_model), ('rf', rf_model), ('xgb', xgb_model)]

        # Create our voting classifier, inputting our models
        ensemble = VotingRegressor(estimators)

        # fit model to training data
        ensemble.fit(X_train, y_train)

        # test our model on the test data
        print(ensemble.score(X_test, y_test))

        prediction = ensemble.predict(X_test)

        return ensemble


    print(data)
    # Separate the data into the training set (January and February) and the testing set (March)
    train_data = data[(data.index.month >= 7) | (data.index.month == 1) | (data.index.month == 2)]
    test_data = data[data.index.month == 3]

    print(test_data)

    # Prepare the training and test data
    X_train = train_data.drop(['close'], axis=1)
    y_train = train_data['close']

    y_test = test_data['close']
    X_test = test_data.drop(['close'], axis=1)

    rf_model = _train_random_forest(X_train, y_train, X_test, y_test)
    svr_model = _train_svr(X_train, y_train, X_test, y_test)
    xgb_model = _train_xgboost(X_train, y_train, X_test, y_test)
    ensemble_model = _ensemble_model(rf_model, svr_model, xgb_model, X_train, y_train, X_test, y_test)

    rf_prediction = rf_model.predict(X_test)
    svr_prediction = svr_model.predict(X_test)
    xgb_prediction = xgb_model.predict(X_test)
    ensemble_prediction = ensemble_model.predict(X_test)

    print('rf prediction is ', rf_prediction)
    print('svr prediction is ', svr_prediction)
    print('ensemble prediction is ', ensemble_prediction)
    print('truth values are ', y_test.values)

    # Calculate the MSE
    mse = mean_squared_error(y_test, rf_prediction)
    mse = mean_squared_error(y_test, rf_prediction)
    mse = mean_squared_error(y_test, rf_prediction)

    # Calculate the RMSE
    rf_rmse = mean_squared_error(y_test, rf_prediction, squared=False)
    svr_rmse = mean_squared_error(y_test, svr_prediction, squared=False)
    xgb_rmse = mean_squared_error(y_test, xgb_prediction, squared=False)
    ensemb_rmse = mean_squared_error(y_test, ensemble_prediction, squared=False)

    print("Root Mean Squared Error (rf_rmse): {:.2f}".format(rf_rmse))
    print("Root Mean Squared Error (svr_rmse): {:.2f}".format(svr_rmse))
    print("Root Mean Squared Error (xgb_rmse): {:.2f}".format(xgb_rmse))
    print("Root Mean Squared Error (ensemb_rmse): {:.2f}".format(ensemb_rmse))

    # Determine which model has the smallest RMSE
    if rf_rmse == min(rf_rmse, svr_rmse, xgb_rmse, ensemb_rmse):
        minerror_pred = rf_prediction
    elif svr_rmse == min(rf_rmse, svr_rmse, xgb_rmse, ensemb_rmse):
        minerror_pred = svr_prediction
    elif xgb_rmse == min(rf_rmse, svr_rmse, xgb_rmse, ensemb_rmse):
        minerror_pred = xgb_prediction
    else:
        minerror_pred = ensemble_prediction

    # Create a DataFrame with actual and predicted values
    march_dates = data[data.index.month == 3].index.date
    actual_pred_df = pd.DataFrame(
        {'Date': march_dates, 'Stock Name': ticker, 'Actual': y_test, 'Prediction': minerror_pred})

    df_all_predicts = df_all_predicts._append(actual_pred_df)

    # Get the actual stock prices for March
    actual_prices = test_data['close'].values

    # Plot the actual prices
    plt.plot(actual_prices, label='Actual Prices')

    # Plot the predicted prices for each model
    plt.plot(rf_prediction, label='Random Forest Predictions')
    plt.plot(svr_prediction, label='SVR Predictions')
    plt.plot(xgb_prediction, label='XGB Predictions')

    plt.plot(ensemble_prediction, label='Ensemble Predictions')

    # Set the title, x-label, and y-label for the plot
    plt.title(f"{ticker}")
    plt.xlabel('Days')
    plt.ylabel('Price')

    # Show the legend for the plot
    plt.legend()

    # Show the plot
    plt.show()

# Rename the index column to "Date"
df_all_predicts = df_all_predicts.rename_axis("Date")
# save the DataFrame as a CSV file
df_all_predicts.to_csv("prediction_results.csv", index=True)

# Rename the index column to "Date"
df_all_companies = df_all_companies.rename_axis("Date")
# save the DataFrame as a CSV file
df_all_companies.to_csv("historical_results.csv", index=True)

