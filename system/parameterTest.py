import pandas as pd
from implementation import run_parameter_test, analyze_and_save_results
import numpy as np

try:
    # Assuming the script is run from the root directory
    data = pd.read_csv("data/showcaseData_cleaned_v2.csv", index_col="timestamp", parse_dates=True)
    print("Data loaded successfully.")
    print("Columns:", data.columns.tolist())
except FileNotFoundError:
    print("Error: showcaseData_cleaned_v2.csv not found. Please ensure the file is in system/data/")
    exit()

# 2. Feature Engineering: Calculate EMA
# We'll use a common period, e.g., 20, as 'ema' is not in the cleaned file.
ema_period = 20
data['ema'] = data['price'].ewm(span=ema_period, adjust=False).mean()
print(f"\nCalculated {ema_period}-period EMA and added it to the dataframe.")

# 3. Define the Entry Signal
# Condition: Price is within 0.05% of the EMA value.
percentage_threshold = 0.0005
entry_signals = (np.abs(data['price'] - data['ema']) / data['ema']) <= percentage_threshold
print(f"Generated entry signals. Found {entry_signals.sum()} signals where price is within {percentage_threshold:.3%} of EMA.")

# 4. Define Columns for the backtest
price_col = 'price'
# We don't have a specific signal strength column for this strategy.
signal_strength_col = None

# 5. Define Parameter Grid for testing
param_grid = {
    'holding_periods': [60, 120, 180, 240],
    'sampling_percentages': [0.5, 0.8, 1.0]
}
print("\nParameter grid for testing:")
print(param_grid)

# 6. General backtesting settings
n_strategies = 100
batch_size = 50
output_dir = "showcase_ema_proximity_test"
print("\nBacktest settings:")
print(f"Number of strategies: {n_strategies}")
print(f"Batch size: {batch_size}")
print(f"Output directory: {output_dir}\n")


# --- Execution ---
print("--- Starting Parameter Test ---")
results_df = run_parameter_test(
    data=data,
    price_col=price_col,
    entry_signals=entry_signals,
    param_grid=param_grid,
    n_strategies=n_strategies,
    batch_size=batch_size,
    signal_strength_col=signal_strength_col
)

print("\n--- Parameter Test Finished ---")

if results_df is not None and not results_df.empty:
    analyze_and_save_results(
        results_df=results_df,
        data=data,
        entry_signals=entry_signals,
        output_dir=output_dir
    )
    print("\n--- Analysis Complete ---")
else:
    print("No results were generated from the parameter test.")
