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


def months_difference(date1, date2):
    '''Calculates the difference in months between two dates.'''
    date1 = re.sub(r"-", "/", date1)
    date2 = re.sub(r"-", "/", date2)
    # Convert string dates to datetime objects
    d1 = datetime.strptime(date1, "%Y/%m/%d")
    d2 = datetime.strptime(date2, "%Y/%m/%d")

    # Calculate the difference in years and months
    year_diff = d2.year - d1.year
    month_diff = d2.month - d1.month

    # Total months difference
    total_months = year_diff * 12 + month_diff

    return total_months


def read_write_data():
    '''Reads the SPARQL output and writes updated data to a file, including the difference in
    months and an accumulation of keywords.
    '''
    # File paths
    input_csv = 'data/trajectory_love.csv'  # Input CSV file path
    output_csv = 'data/data_rep.csv'  # Output CSV file path

    # Initialize a dictionary to hold data grouped by title
    id_data = {}

    # Read the input CSV
    with open(input_csv, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)

        # Group rows by title and accumulate keywords
        for row in reader:
            id = row['id']

            if id not in id_data:
                id_data[id] = {
                    'id': row['id'],
                    'kudos': row['kudos'],
                    'title': row['title'],
                    'romanticCategory': row['romanticCategory'],
                    'rating': row['rating'],
                    'contentWarning': row['contentWarning'],
                    'words': row['words'],
                    'packaged': row['packaged'],
                    'published': row['published'],
                    'keywords': set([row['keyword']])
                }
            else:

                id_data[id]['keywords'].add(row['keyword'])
                
    # Write the result to a new CSV
    with open(output_csv, 'w', newline='', encoding='utf-8') as outfile:
        # Define the fieldnames
        fieldnames = ['id', 'kudos', 'title', 'keywords', 'amount_keywords',
                      'romanticCategory', 'rating', 'contentWarning',
                      'words', 'packaged', 'published', 'up_time']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()

        for id, data in id_data.items():
            writer.writerow({
                'id': data['id'],
                'kudos': data['kudos'],
                'title': data['title'],
                'keywords': ', '.join(data['keywords']), 
                'amount_keywords': len(data['keywords']),
                'romanticCategory': data['romanticCategory'],
                'rating': data['rating'],
                'contentWarning': data['contentWarning'],
                'words': data['words'],
                'packaged': data['packaged'],
                'published': data['published'],
                'up_time': months_difference(data['published'], data['packaged'][:-9])
            })
    pass


def create_tfidf_data(df):
    '''Creates tfidf columns for keywords, rating, contentWarning and romanticCategory'''
    columns_to_process = ['keywords', 'rating', 'contentWarning', 'romanticCategory']  # Update with your column names

    # Choose the TF-idf vectorizer
    vectorizer = TfidfVectorizer()

    for col in columns_to_process:
        documents = df[col].dropna().tolist()  # Also drops NaN values
        documents = [' '.join(doc.split(',')) for doc in documents]

        # Compute TF-IDF for a column
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Convert the TF-IDF matrix to a DataFrame
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

        # Drop rows in the original DataFrame where the column has NaN, to match the TF-IDF result shape
        df_non_na = df[col].dropna().index

        # Overwrite the original column with the sum of the TF-IDF weights per row
        df.loc[df_non_na, col] = tfidf_df.sum(axis=1).values

    df.to_csv('data/tfidf_rep.csv', index=False)
    pass


def print_prediction(prediction, actual):
    '''Prints the predicted value and the actual value of the regression models.'''

    prediction = list(prediction)
    actual = list(actual)

    for i in range(len(prediction)):
        print(f'predicted value: {int(prediction[i])} ' + '\t\t\t' + f'actual value: {actual[i]}')


def main():
    args = create_arg_parser()

    if not os.path.exists("data/data_rep.csv"):
        read_write_data()

    df = pd.read_csv("data/data_rep.csv")

    if not os.path.exists("data/tfidf_rep.csv"):
        create_tfidf_data(df)
    
    tfidf_df = pd.read_csv("data/tfidf_rep.csv")

    X = df.iloc[:, [4, 8, 11]]
    Z = tfidf_df.iloc[:, [5, 6, 7]]

    X = pd.concat([X, Z], axis=1)
    y = df["kudos"]  # Creating a target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler().fit(X_train)
    scaler_test = StandardScaler().fit(X_test)

    X_scaled = scaler.transform(X_train)
    X_scaled_test = scaler_test.transform(X_test)


    if args.algorithm == "dt":
        model = DecisionTreeRegressor(random_state=42, min_samples_leaf=30, splitter="random")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_test = y_test.to_numpy()

        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        
        print("Regression Results for Decision Trees:")
        print()
        print(f'Mean Squared Error: {round(mse, 3)}')
        print(f'Root Mean Squared Error: {round(rmse, 3)}')

        data_pred = np.array([y_pred, y_test])
        dataset = pd.DataFrame({'Predictions': data_pred[0], 'Actual_Values': data_pred[1]})
        dataset.to_csv('data/predictions_DT.csv', index=False)

    if args.algorithm == "rf":
        rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)

        mse_rf = mean_squared_error(y_test, y_pred)
        rmse_rf = root_mean_squared_error(y_test, y_pred)

        print("Regression Results for Random Forest:")
        print()
        print(f'Mean Squared Error: {round(mse_rf, 3)}')
        print(f'Root Mean Squared Error: {round(rmse_rf, 3)}')

        data_pred = np.array([y_pred, y_test])
        dataset = pd.DataFrame({'Predictions': data_pred[0], 'Actual_Values': data_pred[1]})
        dataset.to_csv('data/predictions_RF.csv', index=False)


    if args.algorithm == "lr":
        log_model = LogisticRegression(random_state=42, max_iter = 100)
        log_model.fit(X_scaled, y_train)

        y_pred = log_model.predict(X_scaled_test)

        mse_log = mean_squared_error(y_test, y_pred)
        rmse_log = root_mean_squared_error(y_test, y_pred)

        print("Regression Results for Logistic Regression:")
        print()
        print(f'Mean Squared Error: {round(mse_log, 3)}')
        print(f'Root Mean Squared Error: {round(rmse_log, 3)}')

        data_pred = np.array([y_pred, y_test])
        dataset = pd.DataFrame({'Predictions': data_pred[0], 'Actual_Values': data_pred[1]})
        dataset.to_csv('data/predictions_LR.csv', index=False)

    if args.algorithm == "svr":
        svr_model = SVR()
        svr_model.fit(X_scaled, y_train)

        y_pred = svr_model.predict(X_scaled_test)

        mse_svr = mean_squared_error(y_test, y_pred)
        rmse_svr = root_mean_squared_error(y_test, y_pred)

        print("Regression Results for SVR:")
        print()
        print(f'Mean Squared Error: {round(mse_svr, 3)}')
        print(f'Root Mean Squared Error: {round(rmse_svr, 3)}')

        data_pred = np.array([y_pred, y_test])
        dataset = pd.DataFrame({'Predictions': data_pred[0], 'Actual_Values': data_pred[1]})
        dataset.to_csv('data/predictions_SVR.csv', index=False)

    if args.print_pred:
        print_prediction(y_pred, y_test)

if __name__ == "__main__":
    main()