
# Import required libraries
import pandas as pd
import numpy as np
import json
import os

# Import project modules
import config
from collector import (
    load_data,
    basic_quality_control,
    detect_monthly_outliers,
    create_next_day_target,
    split_data_strict,
    impute_missing_values
)
from features import create_features
from models import train_random_forest_models
from rules import classify_alert
from report import generate_report

# Main agent execution function
def run_agent():

    # Load the original weather dataset
    raw_data = load_data()

    # Apply basic data quality control
    cleaned_data, qc_log = basic_quality_control(raw_data)

    # Detect monthly statistical outliers
    cleaned_data, outlier_report = detect_monthly_outliers(cleaned_data)

    # Construct the next-day rainfall prediction target
    model_data = create_next_day_target(cleaned_data)

    # Split observations chronologically
    train_data, test_data = split_data_strict(model_data)

    # Impute missing weather values using training data only
    train_data, test_data, imputation_log = impute_missing_values(
        train_data,
        test_data
    )

    # Display initial pipeline results
    print("RAIN-GUARD DATA PIPELINE")
    print("=" * 50)
    print("Selected stations:", config.STATIONS)
    print("Random seed:", config.RANDOM_STATE)
    print("Training observations:", len(train_data))
    print("Testing observations:", len(test_data))
    print("Data processing completed.")

    # Train station-specific and pooled Random Forest models
    models = train_random_forest_models(train_data)

    # Store model evaluation results
    evaluation_results = []

    # Evaluate each station-specific model
    for station in config.STATIONS:

        station_test = test_data[
            test_data["Station"] == station
        ]

        X_test, y_test = create_features(station_test)

        predictions = models[station].predict(X_test)

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score
        )

        evaluation_results.append({
            "Model": station,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, zero_division=0),
            "Recall": recall_score(y_test, predictions, zero_division=0),
            "F1-score": f1_score(y_test, predictions, zero_division=0)
        })

    # Evaluate the pooled model
    X_test, y_test = create_features(test_data)
    pooled_predictions = models["POOLED"].predict(X_test)

    evaluation_results.append({
        "Model": "POOLED",
        "Accuracy": accuracy_score(y_test, pooled_predictions),
        "Precision": precision_score(y_test, pooled_predictions, zero_division=0),
        "Recall": recall_score(y_test, pooled_predictions, zero_division=0),
        "F1-score": f1_score(y_test, pooled_predictions, zero_division=0)
    })

    evaluation_df = pd.DataFrame(evaluation_results)

    # Test generalization to a station excluded from model training
    unseen_train = train_data[
        train_data["Station"].isin(["MURREE", "NARAN"])
    ]

    unseen_test = test_data[
        test_data["Station"] == "PLANDRI"
    ]

    X_unseen_train, y_unseen_train = create_features(unseen_train)
    X_unseen_test, y_unseen_test = create_features(unseen_test)

    from sklearn.ensemble import RandomForestClassifier

    unseen_model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        class_weight=config.CLASS_WEIGHT,
        n_jobs=-1
    )

    unseen_model.fit(X_unseen_train, y_unseen_train)
    unseen_predictions = unseen_model.predict(X_unseen_test)

    unseen_results = {
        "Training stations": "MURREE, NARAN",
        "Unseen test station": "PLANDRI",
        "Accuracy": accuracy_score(y_unseen_test, unseen_predictions),
        "Precision": precision_score(y_unseen_test, unseen_predictions, zero_division=0),
        "Recall": recall_score(y_unseen_test, unseen_predictions, zero_division=0),
        "F1-score": f1_score(y_unseen_test, unseen_predictions, zero_division=0)
    }

    print("Model training and evaluation completed.")
    print(evaluation_df)
    print("\nUnseen-station results:")
    print(unseen_results)

    # Calculate annual rainfall totals
    trend_data = cleaned_data.copy()
    trend_data["Year"] = trend_data["Date"].dt.year

    annual_data = (
        trend_data.groupby(["Station", "Year"])
        .agg(
            Annual_Precipitation=("Precipitation", "sum"),
            Observed_Days=("Precipitation", "count")
        )
        .reset_index()
    )

    # Import statistical analysis functions
    from scipy.stats import kendalltau, theilslopes

    trend_results = []

    # Analyse rainfall trends for each station
    for station in config.STATIONS:

        station_annual = annual_data[
            annual_data["Station"] == station
        ].copy()

        # Retain years with at least 90% data coverage
        station_annual = station_annual[
            station_annual["Observed_Days"] >=
            365 * config.ANNUAL_COVERAGE_THRESHOLD
        ]

        years = station_annual["Year"].values
        rainfall = station_annual["Annual_Precipitation"].values

        # Apply Kendall trend test and Sen's slope
        tau, p_value = kendalltau(years, rainfall)
        sen_slope = theilslopes(rainfall, years)[0]

        # Classify the observed trend
        if p_value < 0.05 and sen_slope > 0:
            trend = "Increasing"
        elif p_value < 0.05 and sen_slope < 0:
            trend = "Decreasing"
        else:
            trend = "No significant trend"

        trend_results.append({
            "Station": station,
            "Years analysed": len(years),
            "Kendall's tau": tau,
            "p-value": p_value,
            "Sen's slope (mm/year)": sen_slope,
            "Trend": trend
        })

    trend_results_df = pd.DataFrame(trend_results)

    print("\nRainfall trend analysis completed.")
    print(trend_results_df)

    # Generate rainfall risk alerts
    alert_records = []

    for station in config.STATIONS:

        station_test = test_data[
            test_data["Station"] == station
        ].copy()

        X_station, y_station = create_features(station_test)

        # Calculate next-day rainfall probabilities
        probabilities = models[station].predict_proba(X_station)[:, 1]

        # Classify each prediction
        for index, probability in enumerate(probabilities):

            alert, action = classify_alert(
                station,
                probability,
                config.ALERT_THRESHOLDS
            )

            alert_records.append({
                "Date": station_test.iloc[index]["Date"],
                "Station": station,
                "Rainfall_Probability": probability,
                "Alert_Level": alert,
                "Recommended_Action": action,
                "Actual_Rainfall": int(y_station.iloc[index])
            })

    alerts_df = pd.DataFrame(alert_records)

    # Save the latest available status for each station
    latest_status = {}

    for station in config.STATIONS:

        station_alerts = alerts_df[
            alerts_df["Station"] == station
        ].sort_values("Date")

        latest = station_alerts.iloc[-1]

        latest_status[station] = {
            "Date": latest["Date"].strftime("%Y-%m-%d"),
            "Rainfall_Probability": round(
                float(latest["Rainfall_Probability"]), 3
            ),
            "Alert_Level": latest["Alert_Level"],
            "Recommended_Action": latest["Recommended_Action"]
        }

    # Write the station statuses to a JSON file
    with open("status.json", "w") as file:
        json.dump(latest_status, file, indent=4)

    print("\nAlert generation completed.")
    print(alerts_df["Alert_Level"].value_counts())
    print("\nLatest station statuses:")
    print(json.dumps(latest_status, indent=4))

    # Calculate High-alert performance
    scorecard_records = []

    for station in config.STATIONS:

        station_alerts = alerts_df[
            alerts_df["Station"] == station
        ]

        # Identify High alerts and actual rainfall events
        high_alert = (
            station_alerts["Alert_Level"] == "High"
        )

        actual_rain = (
            station_alerts["Actual_Rainfall"] == 1
        )

        # Calculate confusion matrix components
        TP = int((high_alert & actual_rain).sum())
        FP = int((high_alert & ~actual_rain).sum())
        TN = int((~high_alert & ~actual_rain).sum())
        FN = int((~high_alert & actual_rain).sum())

        # Calculate performance rates
        hit_rate = TP / (TP + FN) if TP + FN else 0
        false_alarm_rate = FP / (FP + TN) if FP + TN else 0
        miss_rate = FN / (TP + FN) if TP + FN else 0

        scorecard_records.append({
            "Station": station,
            "True Positives": TP,
            "False Positives": FP,
            "True Negatives": TN,
            "False Negatives": FN,
            "Hit Rate": hit_rate,
            "False-Alarm Rate": false_alarm_rate,
            "Miss Rate": miss_rate
        })

    # Calculate overall scorecard
    overall = {
        "Station": "OVERALL",
        "True Positives": sum(r["True Positives"] for r in scorecard_records),
        "False Positives": sum(r["False Positives"] for r in scorecard_records),
        "True Negatives": sum(r["True Negatives"] for r in scorecard_records),
        "False Negatives": sum(r["False Negatives"] for r in scorecard_records)
    }

    overall["Hit Rate"] = (
        overall["True Positives"] /
        (overall["True Positives"] + overall["False Negatives"])
    )

    overall["False-Alarm Rate"] = (
        overall["False Positives"] /
        (overall["False Positives"] + overall["True Negatives"])
    )

    overall["Miss Rate"] = (
        overall["False Negatives"] /
        (overall["True Positives"] + overall["False Negatives"])
    )

    scorecard_records.append(overall)

    scorecard_df = pd.DataFrame(scorecard_records)

    print("\nHigh-alert performance scorecard:")
    print(scorecard_df)

    # Generate rainfall risk distribution chart
    import matplotlib.pyplot as plt

    alert_counts = pd.crosstab(
        alerts_df["Station"],
        alerts_df["Alert_Level"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    alert_counts.plot(kind="bar", ax=ax, width=0.75)

    ax.set_title("Rainfall Risk Alert Distribution by Station")
    ax.set_xlabel("Weather Station")
    ax.set_ylabel("Number of Alerts")
    ax.legend(title="Alert Level")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(
        "rainfall_risk_chart.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close(fig)

    # Save quality-control and outlier reports
    qc_log.to_csv("quality_control_log.csv", index=False)
    outlier_report.to_csv("outlier_report.csv", index=False)
    imputation_log.to_csv("imputation_log.csv", index=False)

    # Generate the consolidated analysis report
    generate_report(
        evaluation_df,
        unseen_results,
        trend_results_df,
        scorecard_df
    )

    # Save all generated alerts
    alerts_df.to_csv("rainfall_alerts.csv", index=False)

    # Display final completion message
    print("\n" + "=" * 50)
    print("RAIN-GUARD AGENT EXECUTION COMPLETED")
    print("=" * 50)
    print("Generated outputs:")
    print("- status.json")
    print("- rainfall_risk_chart.png")
    print("- rainfall_alerts.csv")
    print("- quality_control_log.csv")
    print("- outlier_report.csv")
    print("- imputation_log.csv")
    print("- model_evaluation.csv")
    print("- rainfall_trends.csv")
    print("- high_alert_scorecard.csv")


# Run the agent when this script is executed directly
if __name__ == "__main__":
    run_agent()
