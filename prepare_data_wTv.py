#
#
#
#

import re
import csv

from datetime import datetime
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

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


def create_word2vec(df, columns, vector_size=100, window=5, min_count=1):
    '''Takes a list a df and list of columns as input and creates Word2Vec representations for the specified columns'''
    for column in columns:
        tokenized_column = df[column].apply(lambda x: str(x).split())

        model = Word2Vec(sentences=tokenized_column, vector_size=vector_size, window=window, min_count=min_count)

        df[column] = tokenized_column.apply(lambda tokens: model.wv[tokens].mean(axis=0) if tokens else [0] * vector_size)

    return df

def array_to_value(df_transformed):
  '''Takes a df with arrays of number as input and creates mean, max, and min values for the columns specified in the funtion'''
  
  df_transformed['keywords_mean'] = df_transformed['keyword'].apply(np.mean)
  df_transformed['keywords_max'] = df_transformed['keyword'].apply(np.max)
  df_transformed['keywords_min'] = df_transformed['keyword'].apply(np.min)

  df_transformed['rating_mean'] = df_transformed['rating'].apply(np.mean)
  df_transformed['rating_max'] = df_transformed['rating'].apply(np.max)
  df_transformed['rating_min'] = df_transformed['rating'].apply(np.min)

  df_transformed['contentWarning_mean'] = df_transformed['contentWarning'].apply(np.mean)
  df_transformed['contentWarning_max'] = df_transformed['contentWarning'].apply(np.max)
  df_transformed['contentWarning_min'] = df_transformed['contentWarning'].apply(np.min)

  df_transformed['romanticCategory_mean'] = df_transformed['romanticCategory'].apply(np.mean)
  df_transformed['romanticCategory_max'] = df_transformed['romanticCategory'].apply(np.max)
  df_transformed['romanticCategory_min'] = df_transformed['romanticCategory'].apply(np.min)

  df_transformed = df_transformed.iloc[:, 12:]

  return df_transformed

def main():
    df = read_write_data()
    columns_to_transform = ['keyword', 'rating', 'contentWarning', 'romanticCategory']
    df_wTv = create_word2vec(df, columns_to_transform)
    df_wTv = array_to_value(df_wTv)

    X = df.loc[:, ['words', 'amount_keywords', 'up_time']]
    Z = df_wTv

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
    train.to_csv('data/train_wTv.csv', index=False)
    dev.to_csv('data/dev_wTv.csv', index=False)
    test.to_csv('data/test_wTv.csv', index=False)


if __name__ == "__main__":
    main()