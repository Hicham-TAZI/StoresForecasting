# Evaluation and Forecast Saving Functions in src/utils/utils.py

import matplotlib.pyplot as plt
from darts.metrics import rmse
import os
import logging
import matplotlib.dates as mdates
from datetime import timedelta



def eval_model(model, train_scaled, val_scaled, test_scaled, scaler, store_number, experiment_name="default", base_dir="stores_forecasts"):
    """
    Evaluates the model on both validation and test series, and saves the combined plot.
    """
    # Construct the directory path for the experiment
    experiment_dir = os.path.join(base_dir, f"store_{store_number}", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    plot_path = os.path.join(experiment_dir, "model_evaluation.png")

    # Generate predictions
    pred_series_val = model.predict(series=train_scaled, n=len(val_scaled))
    pred_series_test = model.predict(series=train_scaled.append(val_scaled), n=len(test_scaled))

    # Inverse transform the predictions
    pred_series_val_original = scaler.inverse_transform(pred_series_val)
    pred_series_test_original = scaler.inverse_transform(pred_series_test)

    # Inverse transform the training, validation and test series
    train_original = scaler.inverse_transform(train_scaled)
    val_original = scaler.inverse_transform(val_scaled)
    test_original = scaler.inverse_transform(test_scaled)

    # Calculate RMSE for both phases
    val_rmse = rmse(val_original, pred_series_val_original)
    test_rmse = rmse(test_original, pred_series_test_original)

    logging.info(f"Validation RMSE: {val_rmse}, Test RMSE: {test_rmse}")

    # Plotting
    plt.figure(figsize=(12, 6))

    # Assuming your time series are indexed by datetime objects, calculate the plot range for validation and test
    val_start_date = val_original.start_time() + timedelta(days=10)
    test_start_date = test_original.start_time() + timedelta(days=10)

    # You can adjust the vertical position based on the plot's data
    vertical_pos_val = pred_series_val_original.pd_series().max()+10
    vertical_pos_test = pred_series_test_original.pd_series().min()-10

    # Plotting
    plt.figure(figsize=(12, 6))
    train_original[-3*len(val_scaled):].plot(label="Training Data")
    val_original.plot(label="Validation Actual")
    test_original.plot(label="Test Actual")
    pred_series_val_original.plot(label="Validation Forecast")
    pred_series_test_original.plot(label="Test Forecast")

    # Annotating RMSE scores on the plot
    plt.annotate(f'Val RMSE: {val_rmse:.2f}', xy=(mdates.date2num(val_start_date), vertical_pos_val), 
                xytext=(10, 10), textcoords='offset points', arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.5))

    plt.annotate(f'Test RMSE: {test_rmse:.2f}', xy=(mdates.date2num(test_start_date), vertical_pos_test), 
                xytext=(10, -25), textcoords='offset points', arrowprops=dict(arrowstyle="->"),
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.5))

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
