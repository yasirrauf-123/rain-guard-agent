
# Import required libraries
from sklearn.ensemble import RandomForestClassifier

# Import project configuration
import config

# Import feature engineering
from features import create_features


def train_random_forest_models(train_data):

    # Store trained models
    models = {}

    # Train an individual model for each station
    for station in config.STATIONS:

        station_data = train_data[
            train_data["Station"] == station
        ].copy()

        X, y = create_features(station_data)

        model = RandomForestClassifier(
            n_estimators=config.N_ESTIMATORS,
            random_state=config.RANDOM_STATE,
            class_weight=config.CLASS_WEIGHT,
            n_jobs=-1
        )

        model.fit(X, y)
        models[station] = model

    # Train the pooled model
    X_pooled, y_pooled = create_features(train_data)

    pooled_model = RandomForestClassifier(
        n_estimators=config.N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        class_weight=config.CLASS_WEIGHT,
        n_jobs=-1
    )

    pooled_model.fit(X_pooled, y_pooled)
    models["POOLED"] = pooled_model

    return models
