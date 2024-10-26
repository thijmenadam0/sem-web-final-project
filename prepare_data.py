#
#
#
#

import re
import csv

from datetime import datetime
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


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

    df = pd.read_csv(input_csv)

    # Creates a a updated dataframe aggregated by ID,
    # Makes a set of unique keywords that each ID has.
    grouped_df = (
        df.groupby('id').agg({
            'kudos': 'first',
            'title': 'first',
            'romanticCategory': 'first',
            'rating': 'first',
            'contentWarning': 'first',
            'words': 'first',
            'packaged': 'first',
            'published': 'first',
            'keyword': lambda x: ', '.join(set(x))
        }).reset_index())
    
    # Counts the amount of keywords for each item in the dataframe.
    grouped_df['amount_keywords'] = grouped_df['keyword'].str.split().str.len()

    # Calculates the uptime for each item in the dataframe
    up_times = []
    for i in range(len(grouped_df)):
        published = grouped_df['published'][i]
        packaged = grouped_df['packaged'][i][:-9]
        up_times.append(months_difference(published, packaged))
    grouped_df['up_time'] = up_times

    # If you want the CSV dataframe as well
    # grouped_df.to_csv(output_csv, index=False)

    return grouped_df


def create_tfidf_data(df):
    '''Creates tfidf columns for keywords, rating, contentWarning and romanticCategory'''
    columns_to_process = ['keyword', 'rating', 'contentWarning', 'romanticCategory']  # Update with your column names

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

    # If you want to download the tfidf dataframe
    # df.to_csv('data/tfidf_rep.csv', index=False)
    
    return df


def main():
    df = read_write_data()
    tfidf_df = create_tfidf_data(df)

    X = df.loc[:, ['words', 'amount_keywords', 'up_time']]
    Z = tfidf_df.loc[:, ['keyword', 'romanticCategory', 'rating', 'contentWarning']]

    X = pd.concat([X, Z], axis=1)
    y = df["kudos"]

    n = len(X)
    n_train = int(n * 0.7)
    n_dev = int(n * 0.2)
    n_test = n - n_train - n_dev 

    X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, train_size=n_train, random_state=10)
    X_dev, X_test, y_dev, y_test = train_test_split(X_remaining, y_remaining, test_size=n_test, random_state=10)

    train = pd.concat([X_train, y_train], axis=1)
    dev = pd.concat([X_dev, y_dev], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    train.to_csv('data/train.csv', index=False)
    dev.to_csv('data/dev.csv', index=False)
    test.to_csv('data/test.csv', index=False)


if __name__ == "__main__":
    main()