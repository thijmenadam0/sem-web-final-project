#
#
#
#

import re
import argparse

from datetime import datetime
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer


def create_arg_parser():
    '''Creates an argument parser to read the command line arguments.
    This includes subparsers for the different models.
    '''

    parser = argparse.ArgumentParser()

    # Extra options for the models.
    parser.add_argument("-L", "--log_transform", action="store_true",
                        help="Choose whether to log transform and drop outliers from data.")
    parser.add_argument("-v", "--vectorizer", choices=['tfidf', 'w2v', 'mix'], default='tfidf')

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
    df = pd.read_csv('data/trajectory_love.csv')

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

    # If you want the CSV dataframe as well uncomment the following line
    # grouped_df.to_csv('data/data_rep.csv', index=False)

    return grouped_df


def create_tfidf_data(df):
    '''Creates tfidf columns for keywords, rating, contentWarning and romanticCategory'''
    columns_to_process = ['keyword', 'rating', 'contentWarning', 'romanticCategory']

    # prepare_data.py is for the tfidf Vectorizer. prepare_data_wTv.py uses Word2Vec.
    vectorizer = TfidfVectorizer()

    for col in columns_to_process:
        documents = df[col].dropna().tolist()
        documents = [' '.join(doc.split(',')) for doc in documents]

        tfidf_matrix = vectorizer.fit_transform(documents)
        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
        df_non_na = df[col].dropna().index
        df.loc[df_non_na, col] = tfidf_df.sum(axis=1).values

    # If you want to download the tfidf dataframe uncomment the following line
    # df.to_csv('data/tfidf_rep.csv', index=False)
    
    return df


def create_word2vec(df, columns, vector_size=100, window=5, min_count=1):
    '''Takes a list a df and list of columns as input and creates Word2Vec representations for the specified columns'''
    for column in columns:
        tokenized_column = df[column].apply(lambda x: str(x).split())

        model = Word2Vec(sentences=tokenized_column, vector_size=vector_size,
                         window=window, min_count=min_count)

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


def handle_outliers_with_imputer(df, threshold=1.5, impute_strategy='median'):
    """
    Detects outliers based on the IQR method for each column, replaces them with NaN, 
    and fills NaNs using a SimpleImputer.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (float): Multiplier for the IQR range. Default is 1.5.
        impute_strategy (str): Strategy for imputation ('mean', 'median', 'most_frequent', or 'constant').

    Returns:
        pd.DataFrame: DataFrame with outliers handled and NaNs filled.
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_out = df.copy()
    
    # Step 1: Replace outliers with NaN
    for col in df_out.select_dtypes(include=[np.number]):  # Apply only to numeric columns
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        df_out[col] = np.where((df_out[col] < lower_bound) | (df_out[col] > upper_bound), np.nan, df_out[col])
    
    # Step 2: Impute NaN values
    imputer = SimpleImputer(strategy=impute_strategy)
    df_out[df_out.select_dtypes(include=[np.number]).columns] = imputer.fit_transform(df_out.select_dtypes(include=[np.number]))
    
    return df_out


def log_transform(df, columns=None, zero_replacement=0.001):
    """
    Apply log transformation to specified columns of a DataFrame, replacing zeroes with a small positive value.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): Optional list of columns to transform. 
                      If None, all numeric columns will be transformed.
    - zero_replacement (float): Value to replace zero entries with (default is 1).

    Returns:
    - pd.DataFrame: A new DataFrame with log-transformed columns.
    """
    # Select columns to transform: all numeric columns if `columns` is None
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    # Copy the DataFrame to avoid modifying the original one
    df_transformed = df.copy()
    
    # Apply log transformation to each selected column
    for col in columns:
        # Replace zeroes with a small positive value
        df_transformed[col] = df_transformed[col].replace(0, zero_replacement)
        df_transformed[col] = df_transformed[col].apply(lambda x: zero_replacement if x < 0 else x)
        
        # Apply log transformation
        df_transformed[col] = np.log(df_transformed[col])

    return df_transformed


def alter_df(df):

  df = handle_outliers_with_imputer(df)
  df = log_transform(df)

  return df


def main():
    args = create_arg_parser()
    df = read_write_data()
    columns_to_transform = ['keyword', 'rating', 'contentWarning', 'romanticCategory']
    X = df.loc[:, ['words', 'amount_keywords', 'up_time']]

    if args.vectorizer == 'tfidf':
        tfidf_df = create_tfidf_data(df)

        if args.log_transform:
            tfidf_df = alter_df(tfidf_df)

        Z = tfidf_df.loc[:, columns_to_transform]

        if args.log_transform:
            X = alter_df(X)
        X = pd.concat([X, Z], axis=1)

        if args.log_transform:
            y = alter_df(df)
            y = y["kudos"]
        else:
            y = df["kudos"]

        # Calculates a 70/20/10 split.
        n = len(X)
        n_train = int(n * 0.7)
        n_dev = int(n * 0.2)
        n_test = n - n_train - n_dev 

        X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, train_size=n_train, random_state=10)
        X_dev, X_test, y_dev, y_test = train_test_split(X_remaining, y_remaining, test_size=n_test, random_state=10)

        # Concatenates the data so its all in one file.
        train = pd.concat([X_train, y_train], axis=1)
        dev = pd.concat([X_dev, y_dev], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        if args.log_transform:
            train.to_csv('data/train_log.csv', index=False)
            dev.to_csv('data/dev_log.csv', index=False)
            test.to_csv('data/test_log.csv', index=False)

            # Also writes all X and Y to files for the SHAP test.
            X.to_csv('data/all_X_log.csv', index=False)
            y.to_csv('data/all_y_log.csv', index=False)
        
        else:
            train.to_csv('data/train.csv', index=False)
            dev.to_csv('data/dev.csv', index=False)
            test.to_csv('data/test.csv', index=False)

            # Also writes all X and Y to files for the SHAP test.
            X.to_csv('data/all_X.csv', index=False)
            y.to_csv('data/all_y.csv', index=False)

    if args.vectorizer == 'w2v': 
        df_wTv = create_word2vec(df, columns_to_transform)
        df_wTv = array_to_value(df_wTv)

        if args.log_transform:
            df_wTv = alter_df(df_wTv)
        Z = df_wTv

        if args.log_transform:
            X = alter_df(X)

        X = pd.concat([X, Z], axis=1)
        if args.log_transform:
            y = alter_df(df)
            y = y["kudos"]
        else:
            y = df["kudos"]

        # Calculates a 70/20/10 split.
        n = len(X)
        n_train = int(n * 0.7)
        n_dev = int(n * 0.2)
        n_test = n - n_train - n_dev 

        X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, train_size=n_train, random_state=10)
        X_dev, X_test, y_dev, y_test = train_test_split(X_remaining, y_remaining, test_size=n_test, random_state=10)

        train = pd.concat([X_train, y_train], axis=1)
        dev = pd.concat([X_dev, y_dev], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        if args.log_transform:
            train.to_csv('data/train_wTv_log.csv', index=False)
            dev.to_csv('data/dev_wTv_log.csv', index=False)
            test.to_csv('data/test_wTv_log.csv', index=False)
            # Also writes all X and Y to files for the SHAP test.
            X.to_csv('data/all_X_wTv_log.csv', index=False)
            y.to_csv('data/all_y_wTv_log.csv', index=False)

        else:
            train.to_csv('data/train_wTv.csv', index=False)
            dev.to_csv('data/dev_wTv.csv', index=False)
            test.to_csv('data/test_wTv.csv', index=False)

            # Also writes all X and Y to files for the SHAP test.
            X.to_csv('data/all_X_wTv.csv', index=False)
            y.to_csv('data/all_y_wTv.csv', index=False)

    if args.vectorizer =='mix':
        tfidf_df = create_tfidf_data(df)

        if args.log_transform:
            tfidf_df = alter_df(tfidf_df)

        Z = tfidf_df.loc[:, ['romanticCategory', 'rating', 'contentWarning']]

        df_wTv = create_word2vec(df, columns_to_transform)
        df_wTv = array_to_value(df_wTv)

        if args.log_transform:
            df_wTv = alter_df(df_wTv)

        if args.log_transform:
            X = alter_df(X)
        
        X = pd.concat([X, df_wTv, Z], axis=1)

        if args.log_transform:
            y = alter_df(df)
            y = y["kudos"]
        else:
            y = df["kudos"]

        # Also writes all X and Y to files for the SHAP test.
        X.to_csv('data/all_X_mix.csv', index=False)
        y.to_csv('data/all_y_mix.csv', index=False)

        # Calculates a 70/20/10 split.
        n = len(X)
        n_train = int(n * 0.7)
        n_dev = int(n * 0.2)
        n_test = n - n_train - n_dev

        X_train, X_remaining, y_train, y_remaining = train_test_split(X, y, train_size=n_train, random_state=10)
        X_dev, X_test, y_dev, y_test = train_test_split(X_remaining, y_remaining, test_size=n_test, random_state=10)

        train = pd.concat([X_train, y_train], axis=1)
        dev = pd.concat([X_dev, y_dev], axis=1)
        test = pd.concat([X_test, y_test], axis=1)

        if args.log_transform:
            train.to_csv('data/train_mix_log.csv', index=False)
            dev.to_csv('data/dev_mix_log.csv', index=False)
            test.to_csv('data/test_mix_log.csv', index=False)

            # Also writes all X and Y to files for the SHAP test.
            X.to_csv('data/all_X_mix_log.csv', index=False)
            y.to_csv('data/all_y_mix_log.csv', index=False)

        else:
            train.to_csv('data/train_mix.csv', index=False)
            dev.to_csv('data/dev_mix.csv', index=False)
            test.to_csv('data/test_mix.csv', index=False)

            # Also writes all X and Y to files for the SHAP test.
            X.to_csv('data/all_X_mix.csv', index=False)
            y.to_csv('data/all_y_mix.csv', index=False)


if __name__ == "__main__":
    main()