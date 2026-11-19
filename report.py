
# Import required libraries
import pandas as pd


def generate_report(
    evaluation_df,
    unseen_results,
    trend_results_df,
    scorecard_df
):

    # Display model evaluation results
    print("\nRAIN-GUARD MODEL EVALUATION")
    print("=" * 50)
    print(evaluation_df.to_string(index=False))

    # Display unseen-station generalization results
    print("\nUNSEEN-STATION GENERALIZATION")
    print("=" * 50)
    print(pd.DataFrame([unseen_results]).to_string(index=False))

    # Display rainfall trend analysis
    print("\nRAINFALL TREND ANALYSIS")
    print("=" * 50)
    print(trend_results_df.to_string(index=False))

    # Display High-alert scorecard
    print("\nHIGH-ALERT PERFORMANCE SCORECARD")
    print("=" * 50)
    print(scorecard_df.to_string(index=False))

    # Save evaluation tables
    evaluation_df.to_csv("model_evaluation.csv", index=False)
    trend_results_df.to_csv("rainfall_trends.csv", index=False)
    scorecard_df.to_csv("high_alert_scorecard.csv", index=False)

    print("\nReport tables saved successfully.")
