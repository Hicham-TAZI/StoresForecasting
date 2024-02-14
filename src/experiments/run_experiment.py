# File: ForecastProject/src/experiments/run_experiment.py

import argparse
import logging
from datetime import datetime
from src.data_processing.data_loader import load_preprocess_data, load_all_store_numbers
from src.utils.optuna_optimization import optimize_model
from src.models.tcn_model import build_fit_tcn_model
from src.utils.utils import eval_model, save_forecast_to_csv

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [%(levelname)s] - %(name)s : %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logger = logging.getLogger(__name__)

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log_separator(character='-', length=80, msg=None, color=None):
    """Log a separator line with an optional message in the center, now with customizable color."""
    color = color or Colors.OKGREEN  # Default to green if no color is specified
    if msg:
        half_length = (length - len(msg) - 2) // 2
        line = f"{character * half_length} {msg} {character * half_length}"
        # Ensure the line is exactly `length` characters long even if `msg` length is odd
        line = line[:length]
    else:
        line = character * length
    
    colored_line = f"{color}{line}{Colors.ENDC}"  # Apply the chosen color
    logger.info(f"\n{colored_line}\n")


def run(store_number, n_trials):
    log_separator('*', 80, f"Starting Forecast for Store {store_number}", color=Colors.OKCYAN)
    experiment_name = "experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique experiment identifier

    try:
        log_separator(msg=f"Loading Data for Store {store_number}")
        series = load_preprocess_data('data/DailyOrders.csv', '\t', store_number)
        logger.info("Data loaded and preprocessed successfully.")

        log_separator(msg="Hyperparameter Optimization")
        best_params = optimize_model(series, n_trials)
        logger.info(f"Optimization complete. Best parameters: {best_params}")

        log_separator(msg="Model Training")
        best_model, train_scaled, val_scaled, test_scaled, scaler = build_fit_tcn_model(series, **best_params)
        logger.info("Model training complete.")

        log_separator(msg="Model Evaluation")
        eval_model(best_model, train_scaled, val_scaled, test_scaled, scaler, store_number, experiment_name)
        logger.info("Evaluation plots saved.")

        log_separator(msg="Forecasting Future Data")
        forecast = best_model.predict(n=100, series=train_scaled.append(val_scaled))
        # Inverse transform the forecast
        forecast_original = scaler.inverse_transform(forecast)

        save_forecast_to_csv(forecast_original, store_number, experiment_name)
        logger.info("Forecast saved. Experiment concluded successfully.")
        
    except Exception as e:
        logger.error(f"Error during the forecasting experiment for store {store_number}: {e}")
    
    log_separator('*', 80, f"End of Forecast for Store {store_number}", color=Colors.OKCYAN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run forecasting experiment.")
    parser.add_argument("--store_numbers", type=str, help="Store number for forecasting or 'all' for all stores.", required=True)
    parser.add_argument("--n_trials", type=int, help="Number of trials for hyperparameter optimization.", required=True)
    args = parser.parse_args()

    if args.store_numbers.lower() == 'all':
        store_numbers = load_all_store_numbers()  # Ensure this function is correctly implemented to return a list of store numbers
    else:
        store_numbers = [int(n) for n in args.store_numbers.split(',')]  # Allows for comma-separated list of store numbers

    for store_number in store_numbers:
        run(store_number, args.n_trials)
        log_separator('=', 80)  # Separate logs between different store forecasts for clarity
