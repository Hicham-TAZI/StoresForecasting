# Store Forecasting Project

## Description

This project focuses on forecasting sales for multiple stores using historical sales data. It leverages a Temporal Convolutional Network (TCN) model optimized with Optuna for hyperparameter tuning. The project structure is designed for flexibility, allowing for forecasting individual stores or all stores collectively.

## Project Structure

```plaintext
ForecastProject/
├── data/
│   └── DailyOrders.csv
├── src/
│   ├── data_processing/
│   │   ├── data_loader.py
│   ├── experiments/
│   │   └── run_experiment.py
│   ├── models/
│   │   └── tcn_model.py
│   └── utils/
│       ├── optuna_optimization.py
│       └── utils.py
├── README.md
```


## Setup

To set up the project environment:

1. Clone the repository:
   ```sh
   git clone https://github.com/Hicham-TAZI/StoresForecasting
    ```
2. Navigate to the project directory:
   ```sh
   cd ForecastProject
    ```
3. Install required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

To run a forecasting experiment for a specific store:
```sh
python src/experiments/run_experiment.py --store_numbers <store_number> --n_trials <number_of_trials>
```

To run a forecasting experiment for all store:
```sh
python src/experiments/run_experiment.py --store_numbers 'all' --n_trials <number_of_trials>
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.