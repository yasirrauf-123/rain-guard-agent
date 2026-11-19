
# Import required libraries
import pandas as pd
import numpy as np

# Import project configuration
import config


def load_data(file_path="Data_Input.xlsx"):

    # Load the original Excel dataset
    df = pd.read_excel(file_path)

    # Convert dates into datetime format
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Select the three assigned stations
    df = df[df["Station"].isin(config.STATIONS)].copy()

    # Sort observations by station and date
    df = df.sort_values(["Station", "Date"]).reset_index(drop=True)

    # Return selected dataset
    return df


def basic_quality_control(df):

    # Create a copy to preserve the original dataset
    df = df.copy()

    # Create a log to record data-quality changes
    qc_log = []

    # Replace configured temperature placeholders
    for column in ["Max", "Min"]:

        placeholder_count = df[column].isin(
            config.TEMPERATURE_PLACEHOLDERS
        ).sum()

        df.loc[
            df[column].isin(config.TEMPERATURE_PLACEHOLDERS),
            column
        ] = np.nan

        qc_log.append({
            "Variable": column,
            "Change": "Temperature placeholders replaced",
            "Values affected": int(placeholder_count)
        })

    # Replace negative precipitation values
    negative_precipitation = (df["Precipitation"] < config.MIN_PRECIPITATION).sum()

    df.loc[
        df["Precipitation"] < config.MIN_PRECIPITATION,
        "Precipitation"
    ] = np.nan

    qc_log.append({
        "Variable": "Precipitation",
        "Change": "Negative values replaced",
        "Values affected": int(negative_precipitation)
    })

    # Convert the quality-control log into a DataFrame
    qc_log = pd.DataFrame(qc_log)

    # Return cleaned data and quality-control log
    return df, qc_log

def detect_outliers(df):

    # Create a copy of the dataset
    df = df.copy()

    # Create a log for detected outliers
    outlier_log = []

    # Examine each station separately
    for station in config.STATIONS:

        station_index = df["Station"] == station

        # Examine each weather variable
        for column in config.WEATHER_COLUMNS:

            values = df.loc[station_index, column].dropna()

            # Calculate the median
            median_value = values.median()

            # Calculate the median absolute deviation
            mad_value = np.median(np.abs(values - median_value))

            # Avoid division by zero
            if mad_value == 0:
                outlier_count = 0

            else:
                # Calculate robust modified Z-scores
                modified_z = 0.6745 * (
                    values - median_value
                ) / mad_value

                # Count observations exceeding the configured threshold
                outlier_count = (
                    modified_z.abs() > config.MAD_THRESHOLD
                ).sum()

            # Record the results
            outlier_log.append({
                "Station": station,
                "Variable": column,
                "Median": median_value,
                "MAD": mad_value,
                "Outliers detected": int(outlier_count)
            })

    # Convert the log into a DataFrame
    outlier_log = pd.DataFrame(outlier_log)

    # Return original observations and outlier report
    return df, outlier_log


def detect_monthly_outliers(df):

    # Create a copy of the dataset
    df = df.copy()

    # Extract calendar month
    df["Month"] = df["Date"].dt.month

    # Create an outlier report
    outlier_log = []

    # Examine each station
    for station in config.STATIONS:

        station_data = df[df["Station"] == station]

        # Examine each weather variable
        for column in config.WEATHER_COLUMNS:

            # Examine each calendar month separately
            for month in range(1, 13):

                values = station_data.loc[
                    station_data["Month"] == month, column
                ].dropna()

                # For precipitation, examine non-zero rainfall only
                if column == "Precipitation":
                    values = values[values > 0]

                # Calculate median and MAD
                if len(values) < 10:
                    continue

                median_value = values.median()
                mad_value = np.median(
                    np.abs(values - median_value)
                )

                # Detect outliers when MAD is non-zero
                if mad_value == 0:
                    outlier_count = 0
                else:
                    modified_z = (
                        0.6745 * (values - median_value) / mad_value
                    )

                    outlier_count = (
                        modified_z.abs() > config.MAD_THRESHOLD
                    ).sum()

                # Record monthly results
                outlier_log.append({
                    "Station": station,
                    "Variable": column,
                    "Month": month,
                    "Observations": len(values),
                    "Median": median_value,
                    "MAD": mad_value,
                    "Outliers detected": int(outlier_count)
                })

    # Convert results into a DataFrame
    outlier_log = pd.DataFrame(outlier_log)

    # Return data and outlier report
    return df.drop(columns=["Month"]), outlier_log


def split_data(df):

    # Separate training observations
    train_data = df[
        (df["Date"].dt.year >= config.TRAIN_START_YEAR) &
        (df["Date"].dt.year <= config.TRAIN_END_YEAR)
    ].copy()

    # Separate testing observations
    test_data = df[
        (df["Date"].dt.year >= config.TEST_START_YEAR) &
        (df["Date"].dt.year <= config.TEST_END_YEAR)
    ].copy()

    # Return both datasets
    return train_data, test_data


def impute_missing_values(train_data, test_data):

    # Create copies of both datasets
    train_data = train_data.copy()
    test_data = test_data.copy()

    # Create an imputation log
    imputation_log = []

    # Calculate medians using training data only
    for station in config.STATIONS:

        station_train = train_data["Station"] == station
        station_test = test_data["Station"] == station

        for column in config.WEATHER_COLUMNS:

            # Calculate the station-specific training median
            median_value = train_data.loc[
                station_train, column
            ].median()

            # Use the overall training median as a fallback
            if pd.isna(median_value):
                median_value = train_data[column].median()

            # Count missing values before imputation
            train_missing = train_data.loc[
                station_train, column
            ].isna().sum()

            test_missing = test_data.loc[
                station_test, column
            ].isna().sum()

            # Fill missing values using the training median
            train_data.loc[
                station_train, column
            ] = train_data.loc[
                station_train, column
            ].fillna(median_value)

            test_data.loc[
                station_test, column
            ] = test_data.loc[
                station_test, column
            ].fillna(median_value)

            # Record imputation details
            imputation_log.append({
                "Station": station,
                "Variable": column,
                "Training values imputed": int(train_missing),
                "Testing values imputed": int(test_missing),
                "Training median used": round(float(median_value), 3)
            })

    # Return imputed datasets and log
    return train_data, test_data, pd.DataFrame(imputation_log)


def create_next_day_target(df):

    # Create a copy and sort observations chronologically
    df = df.copy()
    df = df.sort_values(["Station", "Date"]).reset_index(drop=True)

    # Obtain the following observation's date and precipitation
    df["Next_Date"] = df.groupby("Station")["Date"].shift(-1)

    df["Next_Day_Precipitation"] = (
        df.groupby("Station")["Precipitation"].shift(-1)
    )

    # Confirm that the following observation is exactly one day later
    consecutive_day = (
        df["Next_Date"] == df["Date"] + pd.Timedelta(days=1)
    )

    # Keep only observations with valid next-day precipitation
    valid_target = (
        consecutive_day &
        df["Next_Day_Precipitation"].notna()
    )

    df = df.loc[valid_target].copy()

    # Construct the next-day rainfall target
    df["Raintomorrow"] = (
        df["Next_Day_Precipitation"] > 0
    ).astype(int)

    # Remove temporary columns
    df = df.drop(
        columns=["Next_Date", "Next_Day_Precipitation"]
    )

    # Return corrected dataset
    return df


def split_data_strict(df):

    # Define the first day of the testing period
    test_start = pd.Timestamp(
        year=config.TEST_START_YEAR,
        month=1,
        day=1
    )

    # Exclude the final training date to prevent boundary leakage
    train_data = df[
        (df["Date"].dt.year >= config.TRAIN_START_YEAR) &
        (df["Date"] < test_start - pd.Timedelta(days=1))
    ].copy()

    # Select testing observations
    test_data = df[
        (df["Date"].dt.year >= config.TEST_START_YEAR) &
        (df["Date"].dt.year <= config.TEST_END_YEAR)
    ].copy()

    # Return training and testing datasets
    return train_data, test_data
