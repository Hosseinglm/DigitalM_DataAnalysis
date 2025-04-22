# Digital Marketing Data Analysis

A data analysis project for analyzing digital marketing and customer data.

## Project Overview

This project analyzes various digital marketing and customer datasets to extract insights and visualize performance metrics.

## Datasets

The project includes the following datasets in the `data/` directory:

- `google_analytics_data.csv`: Web analytics data including sessions, conversions, and revenue
- `customer_data.csv`: Customer demographic and purchase information
- `facebook_ads_data.csv`: Facebook advertising performance metrics
- `instagram_ads_data.csv`: Instagram advertising performance metrics
- `google_ads_data.csv`: Google Ads performance metrics
- `customer_touchpoints.csv`: Customer interaction data across different channels
- `google_search_console_data.csv`: Search performance data from Google Search Console

## Setup

1. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Analysis

To run the data analysis script:

```
python data_analysis.py
```

This will:
1. Load all datasets
2. Perform analysis on Google Analytics data
3. Analyze customer demographics and behavior
4. Compare ad performance across platforms
5. Generate interactive HTML visualizations

## Visualizations

The script generates several interactive Plotly visualizations saved as HTML files:

- `sessions_by_source.html`: Bar chart of sessions by traffic source
- `sessions_transactions_time.html`: Time series of sessions and transactions
- `sessions_by_device.html`: Pie chart of sessions by device category
- `customer_segments.html`: Bar chart of customer segments
- `age_distribution.html`: Histogram of customer age distribution
- `loyalty_tier_analysis.html`: Analysis of spending by loyalty tier
- `platform_conversions.html`: Comparison of conversions across ad platforms
- `platform_roas.html`: Comparison of ROAS across ad platforms
- `daily_conversions.html`: Time series of daily conversions by platform

Open these HTML files in a web browser to interact with the visualizations.
