
# Rain-Guard Agent Configuration

# Selected weather stations
STATIONS = ["MURREE", "NARAN", "PLANDRI"]

# Random seed for reproducibility
RANDOM_STATE = 3752

# Training and testing periods
TRAIN_START_YEAR = 1971
TRAIN_END_YEAR = 2004

TEST_START_YEAR = 2005
TEST_END_YEAR = 2012

# Data quality control settings

# Values treated as temperature placeholders
TEMPERATURE_PLACEHOLDERS = [100, -99.9]

# Robust outlier detection threshold using MAD
MAD_THRESHOLD = 3.5

# Minimum physically valid precipitation
MIN_PRECIPITATION = 0

# Rainfall prediction label
TARGET_COLUMN = "Raintomorrow"

# Weather variables
WEATHER_COLUMNS = ["Precipitation", "Max", "Min"]

print("Rain-Guard configuration created successfully.")

# Rainfall alert action descriptions
ALERT_ACTIONS = {
    "Normal": "Continue routine rainfall monitoring.",
    "Watch": "Monitor updated forecasts and local weather conditions.",
    "Moderate": "Increase monitoring and notify the designated focal person.",
    "High": "Issue an urgent warning for human review and approval."
}

# Random Forest model settings
N_ESTIMATORS = 100
CLASS_WEIGHT = "balanced"

# Annual rainfall trend analysis
ANNUAL_COVERAGE_THRESHOLD = 0.90

# Alert threshold calibration percentiles
ALERT_PERCENTILES = {
    "Watch": 50,
    "Moderate": 75,
    "High": 90
}

# Station-specific alert probability thresholds
ALERT_THRESHOLDS = {
    "MURREE": {
        "Watch": 0.18,
        "Moderate": 0.36,
        "High": 0.55
    },
    "NARAN": {
        "Watch": 0.26,
        "Moderate": 0.57,
        "High": 0.84
    },
    "PLANDRI": {
        "Watch": 0.20,
        "Moderate": 0.42,
        "High": 0.61
    }
}
