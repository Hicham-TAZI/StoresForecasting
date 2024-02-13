# Evaluation and Forecast Saving Functions in src/utils/utils.py

import matplotlib.pyplot as plt
from darts.metrics import rmse
import os
import logging

def eval_model(model, train_scaled, val_scaled, test_scaled, store_number, experiment_name="default", base_dir="stores_forecasts"):
    """
    Evaluates the model on both validation and test series, and saves the combined plot.
    """
    # Construct the directory path for the experiment
    experiment_dir = os.path.join(base_dir, f"store_{store_number}", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    plot_path = os.path.join(experiment_dir, "model_evaluation.png")

    # Generate predictions
    pred_series_val = model.predict(series=train_scaled, n=len(val_scaled), num_samples=10, mc_dropout=True)
    pred_series_test = model.predict(series=train_scaled.append(val_scaled), n=len(test_scaled), num_samples=10, mc_dropout=True)

    # Calculate RMSE for both phases
    val_rmse = rmse(val_scaled, pred_series_val)
    test_rmse = rmse(test_scaled, pred_series_test)

    logging.info(f"Validation RMSE: {val_rmse}, Test RMSE: {test_rmse}")

    # Plotting
    plt.figure(figsize=(12, 6))
    train_scaled[-3*len(val_scaled):].plot(label="Training Data")
    val_scaled.plot(label="Validation Actual")
    test_scaled.plot(label="Test Actual")
    pred_series_val.plot(label="Validation Forecast")
    pred_series_test.plot(label="Test Forecast")
    plt.title(f"Model Evaluation - Store {store_number}")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

    logging.info(f"Evaluation plot saved to {plot_path}")

def save_forecast_to_csv(forecast_series, store_number, experiment_name="default", base_dir="stores_forecasts"):
    """
    Saves the forecast series to a CSV file within the same structured directory as the evaluation plot.
    """
    # Use the same directory as used for saving the plot
    experiment_dir = os.path.join(base_dir, f"store_{store_number}", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)  # Redundant due to eval_model, but ensures directory exists
    file_path = os.path.join(experiment_dir, "forecast.csv")
    
    # Save the forecast DataFrame to CSV
    forecast_df = forecast_series.pd_dataframe()
    forecast_df.to_csv(file_path, index_label="Time")
    logging.info(f"Forecast saved to {file_path}")
