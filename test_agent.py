
# Import required libraries
import unittest
import pandas as pd
import numpy as np
import os
import json

# Import project modules
import config
from collector import basic_quality_control, create_next_day_target, split_data_strict
from features import create_features
from rules import classify_alert


class TestRainGuardAgent(unittest.TestCase):

    # Test 1: Temperature placeholders are removed
    def test_temperature_placeholders(self):
        sample = pd.DataFrame({
            "Max": [25, 100, 30],
            "Min": [10, 15, -99.9],
            "Precipitation": [2, 0, 5]
        })

        cleaned, log = basic_quality_control(sample)

        self.assertTrue(cleaned["Max"].isna().sum() == 1)
        self.assertTrue(cleaned["Min"].isna().sum() == 1)

    # Test 2: Negative precipitation is removed
    def test_negative_precipitation(self):
        sample = pd.DataFrame({
            "Max": [25, 30],
            "Min": [10, 15],
            "Precipitation": [-2, 5]
        })

        cleaned, log = basic_quality_control(sample)

        self.assertTrue(cleaned["Precipitation"].isna().sum() == 1)

    # Test 3: Next-day target is correctly aligned
    def test_next_day_target(self):
        sample = pd.DataFrame({
            "Date": pd.to_datetime([
                "2000-01-01",
                "2000-01-02",
                "2000-01-03"
            ]),
            "Station": ["MURREE"] * 3,
            "Precipitation": [0, 5, 0],
            "Max": [20, 21, 22],
            "Min": [10, 11, 12]
        })

        result = create_next_day_target(sample)

        self.assertEqual(
            result["Raintomorrow"].tolist(),
            [1, 0]
        )

    # Test 4: Training and testing dates do not overlap
    def test_chronological_split(self):
        sample = pd.DataFrame({
            "Date": pd.to_datetime([
                "2004-12-30",
                "2005-01-01"
            ]),
            "Station": ["MURREE", "MURREE"]
        })

        train, test = split_data_strict(sample)

        self.assertTrue(
            train["Date"].max() < test["Date"].min()
        )

    # Test 5: Target is not included among predictors
    def test_no_target_leakage(self):
        sample = pd.DataFrame({
            "Date": pd.to_datetime(["2000-01-01"]),
            "Precipitation": [5],
            "Max": [20],
            "Min": [10],
            "Raintomorrow": [1]
        })

        X, y = create_features(sample)

        self.assertNotIn("Raintomorrow", X.columns)
        self.assertEqual(len(X.columns), 5)

    # Test 6: Alert classification returns a valid level
    def test_alert_classification(self):
        thresholds = {
            "MURREE": {
                "Watch": 0.18,
                "Moderate": 0.36,
                "High": 0.55
            }
        }

        alert, action = classify_alert(
            "MURREE",
            0.70,
            thresholds
        )

        self.assertEqual(alert, "High")
        self.assertTrue(len(action) > 0)

    # Test 7: Status JSON exists and is readable
    def test_status_file(self):
        self.assertTrue(os.path.exists("status.json"))

        with open("status.json", "r") as file:
            status = json.load(file)

        self.assertTrue(
            all(station in status for station in config.STATIONS)
        )


# Run all automated tests
if __name__ == "__main__":
    unittest.main(argv=[""], exit=False, verbosity=2)
