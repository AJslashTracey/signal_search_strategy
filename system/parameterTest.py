import pandas as pd
from implementation import run_parameter_test, analyze_and_save_results
import numpy as np
import vectorbt as vbt

try:
    # Load data and convert timestamps
    data = pd.read_csv("path_to_Data")
    # Convert millisecond timestamps to datetime
    data['timestamp'] = pd.to_datetime(data['open_time'], unit='ms')
    data.set_index('timestamp', inplace=True)
    
    # Use 'close' as the price column
    data['price'] = data['close']
    
    print("Data loaded successfully.")
    print("Columns:", data.columns.tolist())
except FileNotFoundError:
    print("Error: Ensure file is in correct location.")
    exit()

# 2. Feature Engineering: Calculate Moving Averages
fast_period = 10
slow_period = 50
fast_ma = vbt.MA.run(data['close'], fast_period)
slow_ma = vbt.MA.run(data['close'], slow_period)

# 3. Define the Entry Signal
# MA Crossover strategy: Enter when fast MA crosses above slow MA
entry_signals = fast_ma.ma_crossed_above(slow_ma)
print(f"\nGenerated MA crossover signals using:")
print(f"Fast MA period: {fast_period}")
print(f"Slow MA period: {slow_period}")
print(f"Found {entry_signals.sum()} entry signals")

# 4. Define Columns for the backtest
price_col = 'price'
signal_strength_col = None  # Optional: could use MA difference as signal strength

# 5. Define Parameter Grid for testing
param_grid = {
    'holding_periods': [5, 10, 15, 20],  # Testing different holding periods in days
    'sampling_percentages': [0.3, 0.5, 0.8, 1.0]  # Testing different signal sampling rates
}
print("\nParameter grid for testing:")
print(param_grid)

# 6. General backtesting settings
n_strategies = 1000  # Increased to 1000 as requested
batch_size = 100    # Increased batch size for efficiency
output_dir = "btc_ma_crossover_test"
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

# Optional: Run a single backtest with the original strategy for comparison
print("\n--- Running Single Strategy Backtest for Comparison ---")
pf = vbt.Portfolio.from_signals(
    close=data['close'],
    entries=entry_signals,
    exits=None,  # Using holding periods instead of exit signals
    init_cash=100,
    freq='1D',
    fees=0.001,  # 0.1% trading fee
    slippage=0.001  # 0.1% slippage
)

print("\nOriginal Strategy Results:")
print(f"Total Return: {pf.total_return():.2%}")
print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
print(f"Max Drawdown: {pf.max_drawdown():.2%}")
print(f"Win Rate: {pf.win_rate():.2%}")
print(f"Number of Trades: {pf.num_trades()}")
