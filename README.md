# Food Delivery Analytics Dashboard 

## Overview

The Food Delivery Analytics Dashboard is a comprehensive Streamlit application designed to analyze and visualize food delivery service data. The dashboard provides insights into profitability, operations, pricing strategies, demand forecasting, and business recommendations to optimize food delivery services.

## Table of Contents

* Features
* Technical Architecture
* Data Structure
* Dashboard Components
* Analysis Methodology
* Installation and Setup
* Usage Guide
* Customization
* Troubleshooting
* Future Enhancements

## Features

* **Profitability Analysis**: Analyzes delivery profit margins across different dimensions
* **Operational Insights**: Evaluates delivery efficiency and identifies bottlenecks
* **Dynamic Pricing Models**: Compares static vs. dynamic pricing strategies
* **Geographic Visualization**: Maps restaurant and delivery clusters with heatmaps
* **Demand Forecasting**: Predicts future order volume with time-series analysis
* **Customer Segmentation**: Provides segment-specific pricing strategies
* **Business Recommendations**: Offers actionable insights to improve performance

## Technical Architecture

The dashboard is built using:

* **Streamlit**: The core framework for the web application
* **Pandas/NumPy**: For data manipulation and analysis
* **Plotly**: For interactive charts and visualizations
* **Folium**: For interactive maps and geographic visualization
* **Scikit-learn**: For various analytical models
* **Statsmodels**: For time series forecasting

The application follows a modular architecture with separate functions for:

* Data preprocessing
* Metric calculations
* Visualization generation
* Analysis components
* Recommendation engine

## Data Structure

The dashboard works with food delivery data containing the following key fields:

| Field                         | Description                 | Type              |
| ----------------------------- | --------------------------- | ----------------- |
| Order\_ID                     | Unique order identifier     | String            |
| Customer\_ID                  | Customer identifier         | String            |
| Restaurant\_ID                | Restaurant identifier       | String            |
| Delivery\_person\_ID          | Delivery person identifier  | String            |
| Order\_Date                   | Date of order               | Date (DD-MM-YYYY) |
| Time\_Orderd                  | Time of order               | Time (HH\:MM\:SS) |
| Time\_Order\_picked           | Time of pickup              | Time (HH\:MM\:SS) |
| Restaurant\_latitude          | Restaurant latitude         | Float             |
| Restaurant\_longitude         | Restaurant longitude        | Float             |
| Delivery\_location\_latitude  | Delivery location latitude  | Float             |
| Delivery\_location\_longitude | Delivery location longitude | Float             |
| Time\_taken(min)              | Delivery time in minutes    | String/Float      |
| Delivery\_person\_Age         | Age of delivery person      | Integer           |
| Delivery\_person\_Ratings     | Rating of delivery person   | Float             |
| Type\_of\_order               | Category of food ordered    | String            |
| Type\_of\_vehicle             | Vehicle used for delivery   | String            |
| City                          | City of delivery            | String            |
| Weatherconditions             | Weather during delivery     | String            |
| Road\_traffic\_density        | Traffic conditions          | String            |

## Dashboard Components

### 1. Executive Summary

* Key performance metrics
* Top-level insights
* Performance gauge
* Critical recommendations

### 2. Profitability Analysis

* Profitability by various factors (city, vehicle type, weather)
* Hourly and daily profit patterns
* Profit margin analysis
* Key performance indicators

### 3. Operations Analysis

* Delivery route optimization
* Geospatial analysis with interactive maps
* Route efficiency metrics by city and vehicle type
* Driver performance analysis
* Delivery efficiency by various factors

### 4. Pricing Analysis

* Optimal base fee calculation
* Dynamic vs. static pricing comparison
* Price elasticity visualization
* Customer segment pricing strategies
* Segment-specific recommendations

### 5. Demand Forecasting

* Time-series forecasts of future order volume
* Hourly and daily pattern analysis
* Seasonality visualization
* Forecast accuracy metrics
* Forecast insights and interpretation

### 6. Recommendations

* Pricing recommendations
* Operational improvements
* Driver management strategies
* Route optimization suggestions
* Customer-focused tactics

## Analysis Methodology

### Profitability Calculation

**Profit = Revenue - (Fixed Costs + Variable Costs)**

Where:

* Revenue is the base delivery fee
* Fixed costs include base operating expenses
* Variable costs include time and distance-based expenses

### Geographic Clustering

* K-means clustering for identifying restaurant and delivery hubs
* Heatmaps for visualizing density
* Distance calculations for route efficiency

### Pricing Optimization

* Analyzing the price point that maximizes profit
* Considering price elasticity
* Factoring in competitor pricing
* Segment-specific price sensitivity

### Demand Forecasting

* ARIMA/SARIMA time-series models
* Historical pattern analysis
* Seasonal decomposition
* External factor consideration (weather, events)

## Installation and Setup

### Prerequisites

* Python 3.7+
* pip package manager

### Installation Steps

```bash
git clone https://github.com/username/food-delivery-analytics.git
cd food-delivery-analytics
pip install -r requirements.txt
streamlit run foodDeliveryAnalysisApp.py
```

### Requirements.txt

```text
streamlit==1.15.1
pandas==1.3.5
numpy==1.21.6
matplotlib==3.5.1
seaborn==0.11.2
plotly==5.6.0
folium==0.12.1
streamlit-folium==0.6.15
statsmodels==0.13.2
scikit-learn==1.0.2
```

## Usage Guide

### Loading Data

* Prepare your CSV file with the necessary columns
* Upload through the file uploader in the sidebar
* Alternatively, use the sample data provided

### Navigating Tabs

* Use the tabs at the top of the dashboard to navigate between analysis types
* Expand/collapse sections using the arrow icons
* Interact with charts by hovering, zooming, and filtering
* Toggle map layers using the control panel in the top right

### Interpreting Results

* **KPI Cards**: Display key metrics at the top of the dashboard
* **Profit Charts**: Show profitability across different dimensions
* **Maps**: Visualize geographic patterns and hotspots
* **Forecast Charts**: Project future order volumes
* **Recommendation Cards**: Highlight actionable insights

## Customization

### Branding

The dashboard can be customized by modifying the `BRAND_COLORS` dictionary:

```python
BRAND_COLORS = {
    'primary': '#1E88E5',
    'secondary': '#43A047',
    'accent': '#FF6F00',
    'light_bg': '#F5F7FA',
    'dark_text': '#263238',
    # Additional colors...
}
```

### Adding New Analysis

To add new analysis components:

* Create a new function in the script
* Add the visualization or analysis logic
* Create a new tab or section in the main layout
* Call your function to display the results

```python
def my_new_analysis(df):
    # Analysis logic
    return results

# In main layout
with tab_new:
    st.header("My New Analysis")
    results = my_new_analysis(df)
    st.plotly_chart(results)
```

## Troubleshooting

### Common Issues

* **Map Display Problems**

  * **Issue**: Location coordinates are not in proper format
  * **Solution**: Ensure coordinates are numeric and within valid latitude/longitude ranges
  * **Workaround**: Use the demo map functionality for presentation purposes

* **Forecasting Errors**

  * **Issue**: Insufficient data for time series analysis
  * **Solution**: Ensure at least 30 days of historical data

* **Performance Issues**

  * **Issue**: Dashboard runs slowly with large datasets
  * **Solution**: Use data sampling for initial visualization
  * **Solution**: Implement caching with `@st.cache_data` decorator

* **Visualization Rendering Issues**

  * **Issue**: Charts or text not displaying correctly
  * **Solution**: Check for proper HTML formatting in custom components
  * **Solution**: Ensure proper contrast between text and background colors
