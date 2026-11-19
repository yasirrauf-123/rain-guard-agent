
# Import required libraries
import numpy as np
import pandas as pd

# Import project configuration
import config


def create_features(df):

    # Create a copy of the dataset
    df = df.copy()

    # Extract day of year
    day_of_year = df["Date"].dt.dayofyear

    # Create cyclical seasonal features
    df["Season_Sin"] = np.sin(
        2 * np.pi * day_of_year / 365.25
    )

    df["Season_Cos"] = np.cos(
        2 * np.pi * day_of_year / 365.25
    )

    # Define predictor variables
    feature_columns = [
        "Precipitation",
        "Max",
        "Min",
        "Season_Sin",
        "Season_Cos"
    ]

    # Extract model features
    X = df[feature_columns].copy()

    # Extract next-day rainfall labels
    y = df[config.TARGET_COLUMN].copy()

    # Return predictors and target
    return X, y
