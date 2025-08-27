# Signal Search Strategy System

A modular framework for backtesting trading signals and analyzing their performance across different parameter combinations.

## Structure

-   `implementation.py`: Core backtesting engine with parameter testing and analysis functions
-   `parameterTest.py`: User-facing script for configuring and running backtests
-   `data/`: Directory for input datasets

## Usage

1. Add your dataset to `system/data/`
2. Configure `parameterTest.py`:
   - Set data file path
   - Define entry signals
   - Adjust parameter grid
   - Set output directory
3. Run: `python system/parameterTest.py`

## Output

The system generates:
- Performance metrics CSV
- Filtered signals CSV
- Analysis plots (heatmaps, pairplots) 



## Side notes
Inside of the system/results you can see results I got from testing a simple moving average cross over strategy on Bitcoin and Ethereum 



(Reach out to me if you want the data I did not add it to the repo as it would be too )