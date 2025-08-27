from pathlib import Path
import sys
from line_profiler import profile
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
from numba import njit




# Allow relative imports from parent directory
sys.path.append(str(Path(__file__).resolve().parent.parent))



import numpy as np
import pandas as pd
import vectorbt as vbt

# This block should be empty. All data loading and execution
# is handled by `parameterTest.py` or within the `main` block below.

@profile
def run_parameter_test(
    data: pd.DataFrame, 
    price_col: str, 
    entry_signals: pd.Series, 
    param_grid: dict, 
    n_strategies: int, 
    batch_size: int, 
    signal_strength_col: str = None
):
    """
    Runs a parameter test for a given set of entry signals and parameters.
    """
    price = data[price_col]
    index = data.index
    
    # Simple threshold for high pressure only
    all_signal_idx = index[entry_signals]

    # Pre-convert to numpy array for faster indexing
    all_signal_positions = np.where(entry_signals)[0]
    signal_strength = data[signal_strength_col].values if signal_strength_col else None

    # Report on the number of signals being tested
    print(f"Found {len(all_signal_idx)} signals to sample from for {n_strategies} strategies.")

    # Define parameter ranges
    holding_periods = param_grid.get('holding_periods', [120])
    sampling_percentages = param_grid.get('sampling_percentages', [1.0])

    all_results = []

    # Pre-allocate numpy arrays for faster operations
    index_array = np.arange(len(index))

    for holding_period in holding_periods:
        for sampling_pct in sampling_percentages:
            print(f"Testing with holding_period={holding_period}, sampling_pct={sampling_pct}...")
            
            n_signals_to_sample = int(len(all_signal_idx) * sampling_pct)
            if n_signals_to_sample == 0:
                continue

            is_full_sample = (sampling_pct == 1.0)
            
            # Store metrics from all batches to average later
            param_returns = []
            param_sharpes = []
            param_expectancies = []
            param_z_scores = []
            
            num_strategies_to_run = 1 if is_full_sample else n_strategies
            n_batches = 1 if is_full_sample else (n_strategies + batch_size - 1) // batch_size

            if is_full_sample:
                print("  → Full sample, optimizing to run 1 strategy.")
            else:
                print(f"  → Processing {n_strategies} strategies in {n_batches} batches of up to {batch_size}...")

            total_signals_across_strategies = n_signals_to_sample * num_strategies_to_run
            print(f"  → {n_signals_to_sample} signals per strategy, {total_signals_across_strategies:,} total signals across all strategies")

            for i in range(n_batches):
                if not is_full_sample:
                    print(f"    - Running batch {i+1}/{n_batches}...")

                current_batch_size = 1 if is_full_sample else min(batch_size, n_strategies - i * batch_size)

                if is_full_sample:
                    selected_indices_matrix = np.array([all_signal_positions])
                else:
                    selected_indices_matrix = np.array([
                        np.random.choice(all_signal_positions, size=n_signals_to_sample, replace=False)
                        for _ in range(current_batch_size)
                    ])
                
                entry_signals_np = np.zeros((len(index), current_batch_size), dtype=bool)
                
                for strategy_idx in range(current_batch_size):
                    entry_signals_np[selected_indices_matrix[strategy_idx], strategy_idx] = True
                
                vbt_entry_signals = pd.DataFrame(
                    entry_signals_np,
                    index=index,
                    columns=[f'strategy_{i * batch_size + j}' for j in range(current_batch_size)]
                )

                if signal_strength is not None:
                    batch_z_scores = np.array([
                        signal_strength[selected_indices_matrix[j]].mean()
                        for j in range(current_batch_size)
                    ])
                    param_z_scores.extend(batch_z_scores)

                exit_signals_np = np.zeros_like(entry_signals_np)
                if holding_period < len(index):
                    exit_signals_np[holding_period:] = entry_signals_np[:-holding_period]
                
                vbt_exit_signals = pd.DataFrame(
                    exit_signals_np,
                    index=index,
                    columns=vbt_entry_signals.columns
                )

                pf = vbt.Portfolio.from_signals(
                    close=price,
                    entries=vbt_entry_signals,
                    exits=vbt_exit_signals,
                    init_cash=100,
                    freq='1T',
                    fees=0.0003,
                    slippage=0.0001
                )
                
                param_returns.extend(pf.total_return())
                param_sharpes.extend(pf.sharpe_ratio(freq='1T'))
                param_expectancies.extend(pf.trades.expectancy())

            mean_return = np.mean(param_returns)
            mean_sharpe = np.nanmean(param_sharpes)
            mean_expectancy = np.nanmean(param_expectancies)
            mean_z_score = np.mean(param_z_scores) if param_z_scores else np.nan
            print(f"Mean sharp: {mean_sharpe}")

            all_results.append({
                'holding_period': holding_period,
                'sampling_pct': sampling_pct,
                'total_return': mean_return,
                'sharpe': mean_sharpe,
                'expectancy': mean_expectancy,
                'mean_z_score': mean_z_score,
                'signals_per_strategy': n_signals_to_sample,
                'total_signals': total_signals_across_strategies
            })

    # Convert results to a DataFrame for analysis
    results_df = pd.DataFrame(all_results)
    return results_df

def analyze_and_save_results(results_df: pd.DataFrame, data: pd.DataFrame, entry_signals: pd.Series, output_dir: str):
    """
    Analyzes, plots, and saves the backtest results.
    """
    # Print summary statistics with signal count information
    print("\n--- Parameter Test Results ---")
    print(f"Total available signals: {entry_signals.sum()}")
    print("\nResults (averaged across all strategies):")
    print(results_df)

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save results_df as CSV with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"parameter_test_results_{timestamp}.csv"
    results_filepath = output_path / results_filename
    results_df.to_csv(results_filepath, index=False)
    print(f"Results saved to: {results_filepath}")

    # Create CSV file with signal filtered data
    signal_filtered_data = data[entry_signals].copy()
    signal_filename = f"signal_filtered_data_{timestamp}.csv"
    signal_filepath = output_path / signal_filename
    signal_filtered_data.to_csv(signal_filepath)
    print(f"Signal filtered data saved to: {signal_filepath}")
    print(f"Signal filtered data shape: {signal_filtered_data.shape}")

    # Create a pivot table and heatmap for total return
    plt.figure(figsize=(12, 8))
    return_pivot = results_df.pivot(index='holding_period', columns='sampling_pct', values='total_return')
    # Create a custom colormap with stronger reds for negative values
    custom_cmap = sns.diverging_palette(h_neg=0, h_pos=120, s=99, l=40, sep=3, as_cmap=True, center='light')
    sns.heatmap(return_pivot, annot=True, fmt=".3f", cmap=custom_cmap, center=0)
    plt.title('Mean Total Return')
    plt.tight_layout()
    plt.show()

    # Create a pivot table and heatmap for Sharpe ratio
    plt.figure(figsize=(12, 8))
    sharpe_pivot = results_df.pivot(index='holding_period', columns='sampling_pct', values='sharpe')
    sharpe_pivot_clean = sharpe_pivot.fillna(0)  # or use .dropna()

    try:
        # Use same custom colormap for Sharpe ratio
        custom_cmap = sns.diverging_palette(h_neg=0, h_pos=120, s=99, l=40, sep=3, as_cmap=True, center='light')
        sns.heatmap(sharpe_pivot_clean, annot=True, fmt=".3f", cmap=custom_cmap, center=0, cbar_kws={'label': 'Sharpe Ratio'})
        plt.title('Mean Sharpe Ratio')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error generating Sharpe ratio plot: {e}")

    # Create a pivot table and heatmap for signal counts
    plt.figure(figsize=(12, 8))
    signals_pivot = results_df.pivot(index='holding_period', columns='sampling_pct', values='signals_per_strategy')
    # Use sequential green colormap for non-negative values
    sns.heatmap(signals_pivot, annot=True, fmt=".0f", cmap="Greens")
    plt.title('Number of Signals per Strategy')
    plt.tight_layout()
    plt.show()

    if 'mean_z_score' in results_df.columns:
        plt.figure(figsize=(12, 8))
        zscore_pivot = results_df.pivot(index='holding_period', columns='sampling_pct', values='mean_z_score')
        try:
            # Use sequential green colormap for non-negative values
            sns.heatmap(zscore_pivot, annot=True, fmt=".3f", cmap="Greens")
            plt.title('Mean Z-Score of Selected Signals')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error generating Z-score plot: {e}")

    # Create pairplot with error handling
    try:
        plt.figure(figsize=(15, 10))
        results_df_plot = results_df.copy()
        results_df_plot['sampling_pct'] = results_df_plot['sampling_pct'].astype(str)
        
        sns.pairplot(results_df_plot, hue='sampling_pct', diag_kind='kde')
        plt.suptitle('Parameter Analysis Pairplot', y=1.02)
        plt.show()
    except Exception as e:
        print(f"Error generating pairplot: {e}")

def main():
    """
    Main function to configure and run the backtest.
    """
    # --- Configuration ---
    # 1. Load your data
    # Replace this with your actual data loading logic.
    # For example:
    # data = pd.read_csv("path/to/your/data.csv", index_col="timestamp", parse_dates=True)
    
    # As a placeholder, creating a dummy dataframe:
    print("Loading placeholder data. Replace this with your actual data loading.")
    timestamps = pd.to_datetime(pd.date_range('2023-01-01', periods=2000, freq='T'))
    price = 100 + np.random.randn(2000).cumsum() * 0.1
    data = pd.DataFrame({
        'price': price,
        'trend': np.random.choice(['Uptrend', 'Downtrend'], 2000),
        'feature1': np.random.choice(['a', 'b', 'c'], 2000),
        'signal_strength': np.random.rand(2000)
    }, index=timestamps)

    # 2. Define your entry signals
    # This is where you define the logic for your trading signals.
    # It should result in a boolean pandas Series with the same index as your data.
    entry_signals = (data['trend'] == 'Uptrend') & (data['feature1'] == 'b')

    # 3. Define columns
    price_col = 'price'
    signal_strength_col = 'signal_strength'  # Optional, for z-score analysis

    # 4. Define parameter grid for testing
    param_grid = {
        'holding_periods': [60, 120, 240, 300],
        'sampling_percentages': [0.5, 0.8, 1.0]
    }

    # 5. General settings
    n_strategies = 100
    batch_size = 50
    output_dir = "backtest_results"
    
    # --- Execution ---
    results_df = run_parameter_test(
        data=data,
        price_col=price_col,
        entry_signals=entry_signals,
        param_grid=param_grid,
        n_strategies=n_strategies,
        batch_size=batch_size,
        signal_strength_col=signal_strength_col
    )

    analyze_and_save_results(
        results_df=results_df,
        data=data,
        entry_signals=entry_signals,
        output_dir=output_dir
    )

if __name__ == "__main__":
    main()