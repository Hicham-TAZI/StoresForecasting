import pandas as pd
from darts import TimeSeries

def load_preprocess_data(file_path, delimiter, store_number):
    # Load and preprocess the dataset
    df = pd.read_csv(file_path, delimiter=delimiter)
    df = df[df['Storenumber'] == store_number]
    df['date'] = pd.to_datetime(df['Fiscal Transaction Date'])
    df_filled = df.set_index('date').asfreq('D').fillna(method='ffill').reset_index()
    df_filled['Orders'] = df_filled['Orders'].astype('float32')

    # Convert DataFrame to TimeSeries object for Darts
    series = TimeSeries.from_dataframe(df_filled, 'date', 'Orders', freq='D')

    return series


def load_all_store_numbers(file_path='data/DailyOrders.csv', delimiter='\t'):
    """Load all unique store numbers from the dataset."""
    df = pd.read_csv(file_path, delimiter=delimiter)
    return df['Storenumber'].unique().tolist()


