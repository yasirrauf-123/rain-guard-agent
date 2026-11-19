
# Import project configuration
import config


def classify_alert(station, probability, thresholds):

    # Retrieve station-specific thresholds
    station_thresholds = thresholds[station]

    # Assign the highest applicable alert level
    if probability >= station_thresholds["High"]:
        alert = "High"

    elif probability >= station_thresholds["Moderate"]:
        alert = "Moderate"

    elif probability >= station_thresholds["Watch"]:
        alert = "Watch"

    else:
        alert = "Normal"

    # Retrieve the corresponding action
    action = config.ALERT_ACTIONS[alert]

    # Return alert level and recommended action
    return alert, action
