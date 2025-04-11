# data science
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from bayes_opt import BayesianOptimization

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# display
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
sns.set(style='whitegrid', font='Average')

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

# global vars
ROOT = './data/'

# set numpy seed
SEED = 9
np.random.seed(SEED)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_shape_and_nulls(df):
    """
    Display the shape of a DataFrame and the number of null values in each column.

    Args:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # print shape
    print(f'Shape: {df.shape}')

    # check for missing values
    print('Null values:')

    # display null values
    display(df.isnull().sum().to_frame().T)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_unique_vals_and_dtypes(df):
    """
    Print the number of unique values for each column in a DataFrame.
    If a column has fewer than 20 unique values, print those values. Also shows the data type of each column.

    Args:
    - df (pd.DataFrame): The DataFrame to analyze.

    Returns:
    - None
    """

    # iterate over columns
    for col in df.columns:
        # get number of unique values and print
        n = df[col].nunique()
        print(f'"{col}" ({df[col].dtype}) has {n} unique values')

        # if number of unique values is under 20, print the unique values
        if n < 20:
            print(df[col].unique())
        print()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_hist_with_annot(df, col, bins=None, vertical_lines=None, color='blue'):
    """
    Plots a histogram of a column and optionally adds vertical lines with percentage annotations.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.
    - col (str): Column name to plot.
    - bins (int, optional): Number of bins in the histogram. Default is the square root of the number of rows in the DataFrame.
    - vertical_lines (list[int], optional): List of x-values where vertical lines should be drawn. Defaults to None.

    Returns:
    - None
    """

    # default bins (square root of number of rows)
    if bins is None:
        bins = int(np.sqrt(df.shape[0]))

    # get data
    data = df[col]

    # plot histogram
    ax = data.plot(kind='hist', bins=bins, figsize=(18, 9), title=f'{col} Distribution', color=color)

    # add vertical lines
    if vertical_lines:
        # sort vertical lines to ensure correct region division
        vertical_lines = sorted(vertical_lines)
        
        # add vertical lines
        for x in vertical_lines:
            plt.axvline(x=x, color='black', linestyle='dashed', linewidth=2)
        
        # compute percentages for each region
        total_count = len(data)
        prev_x = 0
        for x in vertical_lines + [data.max()]:  # include max value as final boundary
            region_pct = ((data >= prev_x) & (data < x)).sum() / total_count * 100
            plt.text((prev_x + x) / 2, ax.get_ylim()[1] * 0.9, f'{region_pct:.1f}%', 
                     color='black', fontsize=12, ha='center', va='center', 
                     bbox=dict(facecolor='white', alpha=0.8))
            prev_x = x
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_first_last_rows(df, id):
    """
    Display the first and last rows of a DataFrame.

    Args:
    - df (pd.DataFrame): The DataFrame to display.
    - id (int): The number of rows to display from the start and end of the DataFrame.

    Returns:
    - None
    """

    # display first and last rows
    rows = pd.concat([df[df.id == id].head(1), df[df.id == id].tail(1)])
    display(rows)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def create_features(df):
    """
    Aggregate log data.

    Args:
    - df (pd.DataFrame): DataFrame containing the data.

    Returns:
    - features (pd.DataFrame): DataFrame containing aggregated data.
    """

    # convert to polars and sort
    df_pl = pl.from_pandas(df).sort(by=["id", "event_id"])

    # Compute team stats efficiently
    features = (
        df_pl.group_by("id")
        .agg([
            # 1. "start_delay": first "down_time" in the sorted group
            pl.col("down_time").first().alias("start_delay"),
            
            # 2. "tot_time": total of "action_time" across the group
            pl.col("action_time").sum().alias("tot_time"),
            
            # 3. "num_actions": count of rows in the group
            pl.len().alias("num_actions"),
            
            # 4. "mean_action_time": mean of "action_time"
            pl.col("action_time").mean().alias("mean_action_time"),
            
            # 5. "std_action_time": standard deviation of "action_time"
            pl.col("action_time").std().alias("std_action_time"),
            
            # 6. Proportions for each activity category:
            ((pl.col("activity") == "Nonproduction").mean()).alias("prop_nonproduction"),
            ((pl.col("activity") == "Input").mean()).alias("prop_input"),
            ((pl.col("activity") == "Remove/Cut").mean()).alias("prop_remove_cut"),
            ((pl.col("activity") == "Replace").mean()).alias("prop_replace"),
            ((pl.col("activity") == "Move").mean()).alias("prop_move"),
            ((pl.col("activity") == "Paste").mean()).alias("prop_paste"),
            
            # 7. "cursor_retraction": count the number of times "cursor_position" decreases
            pl.col("cursor_position")
                .diff()
                .fill_null(0)
                .lt(0)
                .sum()
                .alias("cursor_retraction"),
            
            # 8. "word_retraction": count the number of times "word_count" decreases
            pl.col("word_count")
                .diff()
                .fill_null(0)
                .lt(0)
                .sum()
                .alias("word_retraction"),
            
            # 9. "final_word_count": get the last "word_count" after sorting
            pl.col("word_count").last().alias("final_word_count"),

            # 10. final word_count squared
            pl.col("word_count").last().pow(2).alias("final_word_count_squared")
        ])
    )

    # convert back to pandas
    return features.to_pandas()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def xgb_cv(max_depth, n_estimators, learning_rate, gamma, min_child_weight, subsample, colsample_bytree, colsample_bylevel, colsample_bynode, X, y):
    """
    Objective function for XGBoost hyperparameter tuning using Bayesian Optimization.

    Args:
    - XGBRegressor params: Hyperparameters for the XGBoost model.
    - X (pd.DataFrame): Feature DataFrame.
    - y (pd.Series): Target variable.

    Returns:
    - scores.mean (float): Mean 10-fold cross-validation score (negative RMSE).
    """

    # define parameters
    params = {
        'max_depth': int(max_depth),
        'n_estimators': int(n_estimators),
        'learning_rate': learning_rate,
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': colsample_bylevel,
        'colsample_bynode': colsample_bynode,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': SEED 
    }

    # create pipeline
    pipeline = Pipeline([('scaler', MinMaxScaler()), ('model', XGBRegressor(**params))])

    # 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
    scores = cross_val_score(pipeline, X, y, cv=kf, scoring='neg_root_mean_squared_error')

    # return mean 10-fold cross-validation score
    return scores.mean()