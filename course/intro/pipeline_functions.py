import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import pearsonr
import statsmodels.api as sm


def plot_scatter(df, x_name, y_name):
    """Return a Plotly scatterplot for two numeric columns."""
    return px.scatter(df, x=x_name, y=y_name)


def filter_data(df, year):
    """Return rows before the selected year."""
    return df[df["Year"] < year]


def calculate_correlation(df, x_col, y_col):
    """Calculate Pearson correlation and p-value."""
    return pearsonr(df[x_col], df[y_col])


def fit_regression(df, x_col, y_col):
    """Fit a simple OLS regression model."""
    x = sm.add_constant(df[[x_col]])
    y = df[y_col]
    return sm.OLS(y, x).fit()


def tyler_viglen():
    """Create the Tyler Vigen spurious-correlation example dataset."""
    array_1 = np.array([
        227,
        238.806,
        220.93,
        220.358,
        216.652,
        198.424,
        198.587,
        201.298,
        198.333,
        196.481,
        197.041,
        189.078,
        183,
        166,
        157,
        154,
        149,
        128,
        88,
        76.9508,
        59.2101,
        40.4265,
        34.1946,
    ])

    array_2 = np.array([
        5.1,
        5,
        4.7,
        4.6,
        4.4,
        4.3,
        4.1,
        4.2,
        4.2,
        4.2,
        4.1,
        4.2,
        4.2,
        3.9,
        3.96973,
        3.58172,
        3.42805,
        3.42852,
        3.22627,
        3.19709,
        3.033,
        2.40567,
        2.72837,
    ])

    years = np.arange(1999, 2022)

    return pd.DataFrame({
        "Year": years,
        "Kerosene": array_1,
        "DivorceRate": array_2,
    })
