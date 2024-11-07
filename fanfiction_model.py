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

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error

np.random.seed(42)
random.seed(42)


def create_arg_parser():
    '''Creates an argument parser to read the command line arguments.
    This includes subparsers for the different models.
    '''

    parser = argparse.ArgumentParser()

    # Extra options for the models.
    parser.add_argument("-p", "--print_pred", action="store_true",
                        help="Also prints the predicted and actual values.")
    parser.add_argument("-e", "--exporter", action='store_true',
                        help="Also exports the predictions to a .csv file")
    parser.add_argument("-t", "--test", action='store_true',
                        help="Makes it so the model also predicts on the test files.")
    parser.add_argument("-v", "--vectorizer", choices=["tfidf", "wTv", "mix"], default="tfidf",
                        help="Choose the vectorized data you want to use, either Word2Vec or Tfidf or a mix of both "
                        "The data should be prepared with either prepare_data files.")
    parser.add_argument("-L", "--log_transformed", action='store_true',
                        help="Choose whether the data used in log transformed or not.")
    
    subparser = parser.add_subparsers(dest="algorithm", required=True,
                                      help="Choose the classifying algorithm to use")

    # Subparser for the SVR
    svr_parser = subparser.add_parser("svr",
                                      help="Use Support Vector Regression as Regression model")
    svr_parser.add_argument("-k", "--kernel", choices=["linear", "poly", "rbf", "sigmoid"], default="rbf",
                           help="Choose the kernel for the SVR")
    svr_parser.add_argument("-d", "--degree", default=3, type=int,
                               help="ONLY FOR POLY KERNEL, changes the degree of the polynomial kernel function")
    svr_parser.add_argument("-g", "--gamma", choices=["scale", "auto"], default="scale",
                           help="Choose the gamma (kernel coefficient for rbf, poly and sigmoid) for the SFR")
    svr_parser.add_argument("-C", "--C", type=float, default=1, 
                            help="Set the regularization parameter")

    # Subparser for the Linear SVR
    svrl_parser = subparser.add_parser("svrl",
                                      help="Use Linear Support Vector Regression as Regression model")
    svrl_parser.add_argument("-l", "--loss", choices=["epsilon_insensitive", "squared_epsilon_insensitive"], default="epsilon_insensitive",
                           help="Choose the kernel for the SVR")
    svrl_parser.add_argument("-t", "--tol", default=1e-4, type=float,
                               help="Sets the tolerance for stopping criteria")
    svrl_parser.add_argument("-mi", "--max_iter", default=1000, type=int,
                           help="Set the max number of iterations to be run")
    svrl_parser.add_argument("-C", "--C", type=float, default=1, 
                            help="Set the regularization parameter")

    # Parent parser containing the overlapping arguments for Decision Tree and Random Forest
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("-md", "--max_depth", default=None, type=int,
                               help="Set the maximum depth of the tree")
    parent_parser.add_argument("-mss", "--min_samples_split", default=2, type=int,
                               help="Set the minimum number of samples required to split an internal node")
    parent_parser.add_argument("-msl", "--min_samples_leaf", default=1, type=int,
                               help="Set the minimum number of samples per leaf node")
    parent_parser.add_argument("-cr", "--criterion", default='squared_error',
                               choices=['squared_error', 'absolute_error', 'friedman_mse', 'poisson'], 
                               help="Set the criterion value.")

    # Subparser for the Decision Trees
    dt_parser = subparser.add_parser("dt", parents=[parent_parser],
                                      help="Use Decision Tree Regression as Regression model")
    dt_parser.add_argument("-s", "--splitter", choices=["best", "random"], default="best",
                             help="Set the strategy used to choose the split of each node")

    # Subparser for the Random Forest
    rf_parser = subparser.add_parser("rf", parents=[parent_parser],
                                      help="Use Random Forest Regression as Regression model")
    rf_parser.add_argument("-ne", "--number_estimators", default=100, type=int,
                           help="Pick the number of estimators for the random forest model.")

    # Subparser for Linear Regression
    lr_parser = subparser.add_parser("lr",
                                     help="Use Linear Regression as Regression model")
    lr_parser.add_argument("-fi", "--fit_intercept", choices=[True, False], default=True,
                           help="Choose whether to calculate the intercept for this model")

    args = parser.parse_args()

    return args


def print_prediction(prediction, actual):
    '''Prints the predicted value and the actual value of the regression models.'''

    prediction = list(prediction)
    actual = list(actual)

    for i in range(len(prediction)):
        print(f'predicted value: {int(prediction[i])} ' + '\t\t\t' + f'actual value: {actual[i]}')


def choose_model(args):
    '''Chooses the model based on the given arguments.
    Returns the model with hyperparameters, a path where the predictions can be saved
    and the name of the algorithm.
    '''

    # The Decision Trees model
    if args.algorithm == 'dt':
        alg_name = "Decision Trees Regression"
        pred_path = 'predictions/dt.csv'
        model = DecisionTreeRegressor(random_state=42, min_samples_leaf=args.min_samples_leaf,
                                      min_samples_split=args.min_samples_split, max_depth=args.max_depth, 
                                      splitter=args.splitter, criterion=args.criterion)

    # The Random Forest models
    if args.algorithm == 'rf':
        alg_name = "Random Forest Regression"
        pred_path = 'predictions/rf.csv'
        model = RandomForestRegressor(random_state=42, n_estimators=args.number_estimators,
                                         min_samples_leaf=args.min_samples_leaf, min_samples_split=args.min_samples_split,
                                         max_depth=args.max_depth, criterion=args.criterion)

    # The Linear Regression model
    if args.algorithm == "lr":
        alg_name = "Linear Regression"
        pred_path = 'predictions/lr.csv'
        model = LinearRegression(fit_intercept=args.fit_intercept)

    # The SVR model
    if args.algorithm == "svr":
        alg_name = "Support Vector Regression"
        pred_path = 'predictions/svr.csv'
        model = SVR(kernel=args.kernel, degree=args.degree,
                        gamma=args.gamma, C=args.C)

    # The Linear SVR model
    if args.algorithm == "svrl":
        alg_name = "Linear Support Vector Regression"
        pred_path = 'predictions/svrl.csv'
        model = LinearSVR(random_state=42, loss=args.loss, tol=args.tol,
                              max_iter=args.max_iter, C=args.C)

    return pred_path, model, alg_name


def main():
    args = create_arg_parser()

    pred_path, model, algorithm_name = choose_model(args)

    # Reads the train/dev/test split; it is 70/20/10 respectively.
    # As seen in either of the prepare_data code.
    if args.vectorizer == "tfidf":
        train = pd.read_csv("data/train.csv")
        dev = pd.read_csv("data/dev.csv")
        test = pd.read_csv("data/test.csv")
    if args.vectorizer == "wTv":
        train = pd.read_csv("data/train_wTv.csv")
        dev = pd.read_csv("data/dev_wTv.csv")
        test = pd.read_csv("data/test_wTv.csv")
    if args.vectorizer == "mix":
        train = pd.read_csv("data/train_mix.csv")
        dev = pd.read_csv("data/dev_mix.csv")
        test = pd.read_csv("data/test_mix.csv")

    if args.vectorizer == "tfidf" and args.log_transformed == True:
        train = pd.read_csv("data/train_log.csv")
        dev = pd.read_csv("data/dev_log.csv")
        test = pd.read_csv("data/test_log.csv")
    if args.vectorizer == "wTv" and args.log_transformed == True:
        train = pd.read_csv("data/train_wTv_log.csv")
        dev = pd.read_csv("data/dev_wTv_log.csv")
        test = pd.read_csv("data/test_wTv_log.csv")
    if args.vectorizer == "mix" and args.log_transformed == True:
        train = pd.read_csv("data/train_mix_log.csv")
        dev = pd.read_csv("data/dev_mix_log.csv")
        test = pd.read_csv("data/test_mix_log.csv")

    # Creates x and y train
    X_train = train.drop("kudos", axis=1)
    y_train = train["kudos"]

    # Creates x and y dev
    X_dev = dev.drop("kudos", axis=1)
    y_dev = dev["kudos"]

    # Creates x and y test
    X_test = test.drop("kudos", axis=1)
    y_test = test["kudos"]

    # Scales the data so it fits the algorithms better.
    scaler = StandardScaler().fit(X_train)
    scaler_dev = StandardScaler().fit(X_dev)
    scaler_test = StandardScaler().fit(X_test)
    X_train = scaler.transform(X_train)
    X_dev = scaler_dev.transform(X_dev)
    X_test = scaler_test.transform(X_test)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_dev)
    y_dev = y_dev.to_numpy()

    if args.log_transformed:
        y_pred_inverted = np.exp(y_pred)
        y_dev_inverted = np.exp(y_dev)

        mse = mean_squared_error(y_dev_inverted, y_pred_inverted)
        rmse = root_mean_squared_error(y_dev_inverted, y_pred_inverted)

    else:
        mse = mean_squared_error(y_dev, y_pred)
        rmse = root_mean_squared_error(y_dev, y_pred)

    print(f'Regression Results for {algorithm_name} on the Development set:')
    print()
    print(f'Mean Squared Error: {round(mse, 3)}')
    print(f'Root Mean Squared Error: {round(rmse, 3)}')

    if args.test:
        y_pred = model.predict(X_test)
        y_test = y_test.to_numpy()
        
        if args.log_transformed:
            y_pred_inverted = np.exp(y_pred)
            y_test_inverted = np.exp(y_test)

            mse = mean_squared_error(y_test_inverted, y_pred_inverted)
            rmse = root_mean_squared_error(y_test_inverted, y_pred_inverted)

        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = root_mean_squared_error(y_test, y_pred)
        
    
        print()
        print(f'Regression Results for {algorithm_name} on the Test set:')
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