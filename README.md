# Signal Search Strategy System

This directory contains a generalized framework for backtesting trading signals and analyzing their performance. The system is designed to be modular, allowing users to easily test different strategies on various datasets without modifying the core logic.

## File Structure

-   `implementation.py`: The core backtesting engine. This file contains the primary functions for running parameter tests and analyzing results. It is designed to be a reusable module and should not be modified for day-to-day use.

-   `parameterTest.py`: The user-facing script for configuring and running a backtest. This is the main entry point for defining a strategy, loading data, and launching a test.

-   `data/`: This directory is intended to store the datasets used for backtesting (e.g., CSV files).

-   `README.md`: This file, providing an overview of the system.

## How to Use

To run a new backtest, follow these steps:

1.  **Place Your Data**: Add your dataset (e.g., in CSV format) to the `system/data/` directory.

2.  **Configure the Test**: Open `system/parameterTest.py` and modify the configuration section.
    -   **Load Data**: Update the file path to point to your dataset.
    -   **Define Entry Signals**: Implement your custom logic to generate a boolean pandas Series that defines when to enter a trade.
    -   **Set Parameters**: Adjust the `param_grid` to specify the holding periods and sampling percentages you wish to test.
    -   **Set Output**: Change the `output_dir` to a descriptive name for your test.

3.  **Run the Script**: Execute the script from the **root directory** of the project:
    ```sh
    python system/parameterTest.py
    ```

4.  **Review Results**: The script will create a new directory (as specified by `output_dir`) containing the following:
    -   A CSV file with the aggregated performance metrics (`parameter_test_results_...csv`).
    -   A CSV file with the data points that triggered an entry signal (`signal_filtered_data_...csv`).
    -   A series of plots (heatmaps and a pairplot) visualizing the results.

## Core Components (`implementation.py`)

This module provides two key functions:

-   `run_parameter_test()`: Iterates through different parameter combinations (holding periods, sampling percentages) and simulates portfolio performance using `vectorbt`. It returns a DataFrame with aggregated metrics like Sharpe ratio, total return, and expectancy.

-   `analyze_and_save_results()`: Takes the results from the parameter test, generates analytical plots (heatmaps, pairplots), and saves all relevant data and metrics to the specified output directory. 