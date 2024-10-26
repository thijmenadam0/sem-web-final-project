#
#
#
#

import re
import csv
import os
import argparse
import random
import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, root_mean_squared_error

np.random.seed(42)
random.seed(42)


def create_arg_parser():
    '''Creates an argument parser to read the command line arguments.
    This includes subparsers for the different models.
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument("-p", "--print_pred", action="store_true",
                        help="Also prints the predicted and actual values.")
    
    parser.add_argument("-e", "--exporter", action='store_true',
                        help="Also exports the predictions to a .csv file")
    
    parser.add_argument("-t", "--test", action='store_true',
                        help="Makes it so the model also predicts on the test files.")
    
    subparser = parser.add_subparsers(dest="algorithm", required=True,
                                      help="Choose the classifying algorithm to use")
    
    svr_parser = subparser.add_parser("svr",
                                      help="Use Support Vector Regression as Regression model")
    
    dt_parser = subparser.add_parser("dt",
                                      help="Use Decision Tree Regression as Regression model")
    
    rf_parser = subparser.add_parser("rf",
                                      help="Use Random Forest Regression as Regression model")
    
    lr_parser = subparser.add_parser("lr",
                                     help="Use Logistic Regression as Regression model")

    args = parser.parse_args()


    return args


def print_prediction(prediction, actual):
    '''Prints the predicted value and the actual value of the regression models.'''

    prediction = list(prediction)
    actual = list(actual)

    for i in range(len(prediction)):
        print(f'predicted value: {int(prediction[i])} ' + '\t\t\t' + f'actual value: {actual[i]}')


def main():
    args = create_arg_parser()

    # Reads the train/dev/test split; it is 70/20/10 respectively.
    train = pd.read_csv("data/train.csv")
    dev = pd.read_csv("data/dev.csv")
    test = pd.read_csv("data/test.csv")

    # Creates x and y train
    X_train = train.drop("kudos", axis=1)
    y_train = train["kudos"]

    # Creates x and y dev
    X_dev = dev.drop("kudos", axis=1)
    y_dev = dev["kudos"]

    # Creates x and y test
    X_test = test.drop("kudos", axis=1)
    y_test = test["kudos"]

    # Creates the scalers and scaled data for Logistic Regression and SVR.
    scaler = StandardScaler().fit(X_train)
    scaler_dev = StandardScaler().fit(X_dev)
    scaler_test = StandardScaler().fit(X_test)

    X_scaled = scaler.transform(X_train)
    X_scaled_dev = scaler_dev.transform(X_dev)
    X_scaled_test = scaler_test.transform(X_test)

    # The Decision Trees model
    if args.algorithm == "dt":
        pred_path = 'predictions/dt.csv'
        model = DecisionTreeRegressor(random_state=42, min_samples_leaf=30, splitter="random")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_dev)
        y_dev = y_dev.to_numpy()

        mse = mean_squared_error(y_dev, y_pred)
        rmse = root_mean_squared_error(y_dev, y_pred)
        
        print("Regression Results for Decision Trees on the Development set:")
        print()
        print(f'Mean Squared Error: {round(mse, 3)}')
        print(f'Root Mean Squared Error: {round(rmse, 3)}')

        if args.test:
            y_pred = model.predict(X_test)
            y_test = y_test.to_numpy()

            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
        
            print()
            print("Regression Results for Decision Trees on the Test set:")
            print()
            print(f'Mean Squared Error: {round(mse, 3)}')
            print(f'Root Mean Squared Error: {round(rmse, 3)}')

    # The Random Forest model
    if args.algorithm == "rf":
        pred_path = 'predictions/rf.csv'
        rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
        rf_model.fit(X_train, y_train)

        y_pred = rf_model.predict(X_dev)
        y_dev = y_dev.to_numpy()

        mse = mean_squared_error(y_dev, y_pred)
        rmse = root_mean_squared_error(y_dev, y_pred)
        
        print("Regression Results for Random Forest on the Development set:")
        print()
        print(f'Mean Squared Error: {round(mse, 3)}')
        print(f'Root Mean Squared Error: {round(rmse, 3)}')

        if args.test:
            y_pred = rf_model.predict(X_test)
            y_test = y_test.to_numpy()

            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)

            print()
            print("Regression Results for Random Forest on the Test set:")
            print()
            print(f'Mean Squared Error: {round(mse, 3)}')
            print(f'Root Mean Squared Error: {round(rmse, 3)}')

    # The Logistic Regression model
    if args.algorithm == "lr":
        pred_path = 'predictions/lr.csv'
        log_model = LogisticRegression(random_state=42, max_iter = 100)
        log_model.fit(X_scaled, y_train)

        y_pred = log_model.predict(X_scaled_dev)
        y_dev = y_dev.to_numpy()

        mse = mean_squared_error(y_dev, y_pred)
        rmse = root_mean_squared_error(y_dev, y_pred)
        
        print("Regression Results for Logistic Regression on the Development set:")
        print()
        print(f'Mean Squared Error: {round(mse, 3)}')
        print(f'Root Mean Squared Error: {round(rmse, 3)}')

        if args.test:
            y_pred = log_model.predict(X_scaled_test)
            y_test = y_test.to_numpy()

            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)

            print()
            print("Regression Results for Logistic Regression on the Test set:")
            print()
            print(f'Mean Squared Error: {round(mse, 3)}')
            print(f'Root Mean Squared Error: {round(rmse, 3)}')

    # The SVR model.
    if args.algorithm == "svr":
        pred_path = 'predictions/svr.csv'
        svr_model = SVR()
        svr_model.fit(X_scaled, y_train)

        y_pred = svr_model.predict(X_scaled_dev)
        y_dev = y_dev.to_numpy()

        mse = mean_squared_error(y_dev, y_pred)
        rmse = root_mean_squared_error(y_dev, y_pred)

        print("Regression Results for SVR on the Development set:")
        print()
        print(f'Mean Squared Error: {round(mse, 3)}')
        print(f'Root Mean Squared Error: {round(rmse, 3)}')

        if args.test:
            y_pred = svr_model.predict(X_scaled_test)
            y_test = y_test.to_numpy()

            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)

            print()
            print("Regression Results for SVR on the Test set:")
            print()
            print(f'Mean Squared Error: {round(mse, 3)}')
            print(f'Root Mean Squared Error: {round(rmse, 3)}')


    # Prints the predictions based on the test set, if the test set is not given
    # Uses the dev set.
    if args.print_pred:
        if args.test:
            print_prediction(y_pred, y_test)
        else:
            print_prediction(y_pred, y_dev)

    # Exports the predictions if asked, goes to predictions/[filename].csv
    if args.exporter:
        if args.test:
            data_pred = np.array([y_pred, y_test])
        else: 
            data_pred = np.array([y_pred, y_dev])

        dataset = pd.DataFrame({'Predictions': data_pred[0], 'Actual_Values': data_pred[1]})
        dataset.to_csv(pred_path, index=False)

if __name__ == "__main__":
    main()