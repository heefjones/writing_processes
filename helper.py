# data science
import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import normaltest


# algorithms
import math
import itertools

# machine learning
from sklearn.base import is_classifier, clone
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import root_mean_squared_error, r2_score, log_loss, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from xgboost import XGBRegressor, XGBClassifier
import torch
import torch.nn as nn
import torch.optim as optim
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




