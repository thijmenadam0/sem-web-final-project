#
#
#
#

import re
import csv
import os
import random
import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, root_mean_squared_error

np.random.seed(42)
random.seed(42)


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


def keyword_counter(df):
    '''Counts the amount of keywords per item'''
    keywords_list = ["Enemies to Lovers", "Friends to Lovers", "Rivals to Lovers", "Eventual Romance",
                     "enemies to lovers", "Friends to Enemies to Lovers"]

    # Initialize a dictionary to store the counts for each keyword
    keyword_counts = {keyword: 0 for keyword in keywords_list}

    # Go through the 'keywords' column and count occurrences of each keyword
    for entry in df['keywords']:
        for keyword in keywords_list:
            keyword_counts[keyword] += entry.count(keyword)
    
    print(keyword_counts)

    return keyword_counts


def main():
    if not os.path.exists("data/data_rep.csv"):
        read_write_data()

    df = pd.read_csv("data/data_rep.csv")

    if not os.path.exists("data/tfidf_rep.csv"):
        create_tfidf_data(df)
    
    tfidf_df = pd.read_csv("data/tfidf_rep.csv")

    # keyword_counter(df)

    X = df.iloc[:, [4, 8, 11]]
    Z = tfidf_df.iloc[:, [5, 6, 7]]

    print(X, Z)

    X = pd.concat([X, Z], axis=1)
    y = df["kudos"]  # Creating a target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor(random_state=42, min_samples_leaf=30, splitter="random")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_test = y_test.to_numpy()

    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    # for i in range(len(y_test)):
    #    print('Actual value:', y_test[i], 'Predicted value:', y_pred[i], "difference:", y_test[i] - y_pred[i])
    
    print("Decision Trees:")
    print(f'Mean Squared Error: {round(mse, 3)}')
    print(f'Root Mean Squared Error: {round(rmse, 3)}')
    print()

    rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    mse_rf = mean_squared_error(y_test, y_pred_rf)
    rmse_rf = root_mean_squared_error(y_test, y_pred_rf)

    print("Random Forest:")
    print(f'Mean Squared Error: {round(mse_rf, 3)}')
    print(f'Root Mean Squared Error: {round(rmse_rf, 3)}')



if __name__ == "__main__":
    main()