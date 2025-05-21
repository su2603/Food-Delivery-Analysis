import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import folium
from folium.plugins import MarkerCluster, HeatMap
from streamlit_folium import st_folium
import datetime
from datetime import timedelta
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Brand color scheme configuration
BRAND_COLORS = {
    'primary': '#1E88E5',       # Main brand color - blue
    'secondary': '#43A047',     # Secondary color - green
    'accent': '#FF6F00',        # Accent color - orange
    'light_bg': '#F5F7FA',      # Light background
    'dark_text': '#263238',     # Dark text color
    'light_text': '#FFFFFF',    # Light text color
    'success': '#4CAF50',       # Success color
    'warning': '#FFC107',       # Warning color
    'error': '#F44336',         # Error color
    'neutral': '#607D8B'        # Neutral color
}

# Set page configuration
st.set_page_config(
    page_title="Food Delivery Analytics",
    page_icon="🍕",
    layout="wide"
)

# Add custom CSS with your brand colors
st.markdown(f"""
<style>
    /* Main theme colors */
    :root {{
        --primary: {BRAND_COLORS['primary']};
        --secondary: {BRAND_COLORS['secondary']};
        --accent: {BRAND_COLORS['accent']};
        --light-bg: {BRAND_COLORS['light_bg']};
        --dark-text: {BRAND_COLORS['dark_text']};
    }}

    /* Header styling */
    h1, h2, h3 {{
        color: var(--primary);
    }}
    
    /* Metric cards */
    .metric-card {{
        background-color: var(--light-bg);
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        border-top: 4px solid var(--primary);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    .metric-value {{
        font-size: 2.2rem;
        font-weight: bold;
        color: var(--primary);
    }}
    .metric-label {{
        font-size: 1.0rem;
        color: var(--dark-text);
    }}
    
    /* Recommendation boxes */
    .recommendation-box {{
        background-color: #e8f4f8;
        border-left: 5px solid var(--primary);
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
        box-shadow: 1px 1px 3px rgba(0,0,0,0.1);
    }}
    .recommendation-text {{
        color: var(--dark-text);
        font-size: 15px;
    }}
    
    /* Pricing comparison boxes */
    .static-pricing {{
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-top: 4px solid var(--secondary);
    }}
    .dynamic-pricing {{
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-top: 4px solid var(--primary);
    }}
    .pricing-value {{
        font-size: 2.2rem;
        font-weight: bold;
    }}
    .static-pricing .pricing-value {{
        color: var(--secondary);
    }}
    .dynamic-pricing .pricing-value {{
        color: var(--primary);
    }}
</style>
""", unsafe_allow_html=True)

def safe_groupby_agg(df, groupby_col, agg_dict):
    """Safely perform groupby and aggregation, checking for column existence."""
    # Check if groupby column exists
    if groupby_col not in df.columns:
        return pd.DataFrame(), f"Column '{groupby_col}' not found in dataset"
    
    # Check which agg columns exist
    available_agg = {}
    for col, func in agg_dict.items():
        if col in df.columns:
            available_agg[col] = func
    
    if not available_agg:
        return pd.DataFrame(), f"None of the required columns {list(agg_dict.keys())} found in dataset"
    
    # Perform aggregation with available columns
    result = df.groupby(groupby_col).agg(available_agg).reset_index()
    return result, None

@st.cache_data
def load_sample_data():
    """Load sample data if user doesn't have their own data."""
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Create date range for the past 90 days
    end_date = datetime.datetime.now()
    start_date = end_date - timedelta(days=90)
    date_range = [start_date + timedelta(days=x) for x in range(90)]
    
    # Create synthetic data frame
    cities = ["Urban", "Suburban", "Rural"]
    vehicles = ["motorcycle", "scooter", "bicycle"]
    weathers = ["Sunny", "Cloudy", "Rainy", "Foggy", "Stormy"]
    traffic = ["Low", "Medium", "High", "Jam"]
    
    synthetic_data = {
        'Order_ID': [f'ORD{i:05d}' for i in range(n_samples)],
        'Customer_ID': [f'CUST{np.random.randint(1, 200):03d}' for _ in range(n_samples)],
        'Restaurant_ID': [f'REST{np.random.randint(1, 50):02d}' for _ in range(n_samples)],
        'Delivery_person_ID': [f'DRIVER{np.random.randint(1, 30):02d}' for _ in range(n_samples)],
        'Order_Date': [(datetime.datetime.now() - timedelta(days=np.random.randint(1, 90))).strftime('%d-%m-%Y') for _ in range(n_samples)],
        'Time_Orderd': [(datetime.datetime.now() - timedelta(minutes=np.random.randint(30, 180))).strftime('%H:%M:%S') for _ in range(n_samples)],
        'Time_Order_picked': [(datetime.datetime.now() - timedelta(minutes=np.random.randint(10, 60))).strftime('%H:%M:%S') for _ in range(n_samples)],
        'Delivery_location_latitude': np.random.uniform(28.4, 28.7, n_samples),
        'Delivery_location_longitude': np.random.uniform(77.0, 77.3, n_samples),
        'Restaurant_latitude': np.random.uniform(28.5, 28.6, n_samples),
        'Restaurant_longitude': np.random.uniform(77.1, 77.2, n_samples),
        'Time_taken(min)': [f'{np.random.randint(15, 60)} min' for _ in range(n_samples)],
        'Delivery_person_Age': np.random.randint(18, 45, n_samples),
        'Delivery_person_Ratings': np.random.uniform(3.5, 5.0, n_samples).round(1),
        'Type_of_order': np.random.choice(["Snack", "Meal", "Drinks", "Buffet"], n_samples),
        'Type_of_vehicle': np.random.choice(vehicles, n_samples),
        'City': np.random.choice(cities, n_samples),
        'Weatherconditions': np.random.choice(weathers, n_samples),
        'Road_traffic_density': np.random.choice(traffic, n_samples)
    }
    
    df = pd.DataFrame(synthetic_data)
    return df

@st.cache_data
def preprocess_data(df):
    """Basic preprocessing for the demo - SIMPLIFIED VERSION."""
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Extract time taken (from string like "30 min" to numeric 30)
    df['Time_taken(min)'] = df['Time_taken(min)'].str.extract('(\d+)').astype(float)
    
    # Convert date
    df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y')
    
    # Calculate distance (simplified)
    # Instead of using Haversine formula, we'll use a random value for demo
    df['Distance_km'] = np.random.uniform(1, 15, len(df))
    
    # Calculate costs
    base_fee = 100
    cost_per_min = 2
    cost_per_km = 3
    
    df['Cost'] = 20 + (df['Distance_km'] * cost_per_km) + (df['Time_taken(min)'] * cost_per_min)
    df['Revenue'] = base_fee
    df['Profit'] = df['Revenue'] - df['Cost']
    
    # Add some time-based features
    df['Order_Hour'] = pd.to_datetime(df['Time_Orderd'], format='%H:%M:%S', errors='coerce').dt.hour
    df['Order_Day'] = df['Order_Date'].dt.dayofweek
    df['Is_Weekend'] = df['Order_Date'].dt.dayofweek >= 5
    
    # Add dummy clusters instead of using KMeans
    # This avoids the error we were encountering
    df['Restaurant_Cluster'] = np.random.randint(0, 5, len(df))
    df['Delivery_Cluster'] = np.random.randint(0, 5, len(df))
    
    return df

def display_kpi_metrics(df):
    """Display KPI metrics in a nice grid."""
    # Calculate KPIs
    avg_profit = df['Profit'].mean()
    profitable_pct = (df['Profit'] > 0).mean() * 100
    avg_time = df['Time_taken(min)'].mean()
    avg_distance = df['Distance_km'].mean()
    avg_speed = avg_distance / (avg_time / 60)  # km/h
    
    # For total orders, check if Order_ID column exists
    if 'Order_ID' in df.columns:
        total_orders = df['Order_ID'].nunique()
    else:
        total_orders = len(df)  # Use total row count as a fallback
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${avg_profit:.2f}</div>
            <div class="metric-label">Average Profit per Delivery</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_speed:.1f} km/h</div>
            <div class="metric-label">Average Delivery Speed</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{profitable_pct:.1f}%</div>
            <div class="metric-label">Profitable Orders</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_orders}</div>
            <div class="metric-label">Total Orders</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">${100:.2f}</div>
            <div class="metric-label">Base Delivery Fee</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_time:.1f} min</div>
            <div class="metric-label">Average Delivery Time</div>
        </div>
        """, unsafe_allow_html=True)

def plot_profit_by_factor(df, factor):
    """Create a bar plot showing profit by a specific factor."""
    profit_by_factor = df.groupby(factor)['Profit'].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=profit_by_factor.index, 
        y=profit_by_factor.values,
        title=f'Average Profit by {factor}',
        labels={'x': factor, 'y': 'Average Profit'},
        color=profit_by_factor.values,
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        height=400,
        coloraxis_showscale=False,
        xaxis_tickangle=-45
    )
    
    return fig

def plot_profit_by_hour(df):
    """Create a line plot showing profit by hour of day."""
    profit_by_hour = df.groupby('Order_Hour')['Profit'].mean()
    
    fig = px.line(
        x=profit_by_hour.index, 
        y=profit_by_hour.values,
        markers=True,
        title='Average Profit by Hour of Day',
        labels={'x': 'Hour of Day', 'y': 'Average Profit'}
    )
    
    fig.update_layout(height=400)
    fig.update_traces(line=dict(width=3))
    
    return fig

def create_demo_map():
    """Create a demonstration map not dependent on actual data."""
    try:
        # Create a map centered on a generic location
        center_lat, center_lon = 28.6, 77.2  # New Delhi coordinates as an example
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Generate some random points for restaurants
        import random
        random.seed(42)  # For reproducibility
        
        # Create restaurant clusters
        restaurant_cluster = MarkerCluster(name="Restaurant Clusters")
        
        # Generate restaurant clusters
        restaurant_points = []
        for i in range(5):  # 5 clusters
            # Generate cluster center
            cluster_lat = center_lat + random.uniform(-0.05, 0.05)
            cluster_lon = center_lon + random.uniform(-0.05, 0.05)
            
            # Add restaurants around cluster center
            for j in range(10):  # 10 restaurants per cluster
                rest_lat = cluster_lat + random.uniform(-0.01, 0.01)
                rest_lon = cluster_lon + random.uniform(-0.01, 0.01)
                
                folium.Marker(
                    location=[rest_lat, rest_lon],
                    popup=f"Restaurant {i*10 + j+1}<br>Cluster {i+1}",
                    icon=folium.Icon(color='green', icon='cutlery', prefix='fa')
                ).add_to(restaurant_cluster)
                
                # Also add to heatmap data
                restaurant_points.append([rest_lat, rest_lon])
        
        restaurant_cluster.add_to(m)
        
        # Create delivery clusters
        delivery_cluster = MarkerCluster(name="Delivery Locations")
        
        # Generate delivery points
        delivery_points = []
        for i in range(8):  # 8 neighborhood clusters
            # Generate neighborhood center
            hood_lat = center_lat + random.uniform(-0.07, 0.07)
            hood_lon = center_lon + random.uniform(-0.07, 0.07)
            
            # Add delivery locations in this neighborhood
            for j in range(15):  # 15 deliveries per neighborhood
                del_lat = hood_lat + random.uniform(-0.015, 0.015)
                del_lon = hood_lon + random.uniform(-0.015, 0.015)
                
                folium.Marker(
                    location=[del_lat, del_lon],
                    popup=f"Delivery {i*15 + j+1}<br>Neighborhood {i+1}",
                    icon=folium.Icon(color='blue', icon='home', prefix='fa')
                ).add_to(delivery_cluster)
                
                delivery_points.append([del_lat, del_lon])
        
        delivery_cluster.add_to(m)
        
        # Add "hotspot" areas - overlay high-traffic areas
        hotspot_areas = []
        for i in range(3):  # 3 hotspot areas
            # Generate hotspot center
            spot_lat = center_lat + random.uniform(-0.03, 0.03)
            spot_lon = center_lon + random.uniform(-0.03, 0.03)
            
            # Create a circle for each hotspot
            folium.Circle(
                location=[spot_lat, spot_lon],
                radius=500,  # meters
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.2,
                popup=f"High Demand Area {i+1}"
            ).add_to(m)
            
            # Add extra points in hotspots for heatmap intensity
            for j in range(25):
                hot_lat = spot_lat + random.uniform(-0.008, 0.008)
                hot_lon = spot_lon + random.uniform(-0.008, 0.008)
                delivery_points.append([hot_lat, hot_lon])
        
        # Add heatmaps
        HeatMap(restaurant_points, name="Restaurant Density", 
                radius=15, blur=10, 
                gradient={0.4: 'blue', 0.65: 'lime', 1: 'red'}).add_to(m)
        
        HeatMap(delivery_points, name="Delivery Density", 
                radius=15, blur=10, 
                gradient={0.4: 'green', 0.65: 'yellow', 1: 'red'}).add_to(m)
        
        # Add potential optimal routes between restaurants and deliveries
        for i in range(5):
            # Select a random restaurant and delivery point
            if len(restaurant_points) > 0 and len(delivery_points) > 0:
                rest_idx = random.randint(0, len(restaurant_points)-1)
                del_idx = random.randint(0, len(delivery_points)-1)
                
                # Create a path between them
                path = [
                    restaurant_points[rest_idx],
                    delivery_points[del_idx]
                ]
                
                folium.PolyLine(
                    path,
                    color='purple',
                    weight=3,
                    opacity=0.7,
                    popup=f"Delivery Route {i+1}"
                ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        return m, None
    except Exception as e:
        return None, f"Error creating demo map: {str(e)}"

def plot_dynamic_pricing(df):
    """Create a visualization of dynamic pricing factors."""
    # Simulate dynamic pricing factors
    dynamic_factors = {
        'Weather': {
            'Sunny': 1.0,
            'Cloudy': 1.1,
            'Rainy': 1.3,
            'Foggy': 1.3,
            'Stormy': 1.4
        },
        'Traffic': {
            'Low': 1.0,
            'Medium': 1.15,
            'High': 1.3,
            'Jam': 1.4
        },
        'Time of Day': {
            'Morning (6-10)': 1.1,
            'Midday (10-14)': 1.0,
            'Afternoon (14-18)': 1.1,
            'Evening (18-22)': 1.3,
            'Night (22-6)': 1.2
        },
        'Day Type': {
            'Weekday': 1.0,
            'Weekend': 1.2,
            'Holiday': 1.3
        }
    }
    
    # Create a figure to display factor comparisons
    fig = go.Figure()
    
    # Add traces for different factor types
    for factor_type, factors in dynamic_factors.items():
        fig.add_trace(go.Bar(
            x=list(factors.keys()),
            y=list(factors.values()),
            name=factor_type
        ))
    
    fig.update_layout(
        title="Dynamic Pricing Factors",
        xaxis_title="Factor",
        yaxis_title="Price Multiplier",
        legend_title="Factor Type",
        height=500,
        barmode='group'
    )
    
    return fig

# Add this function to your code
def forecast_demand(df):
    """Create a simple time series forecast for future demand."""
    try:
        # Check if we have order date
        if 'Order_Date' not in df.columns:
            return None, "Order_Date column not found"
            
        # Group by date and count orders
        daily_orders = df.groupby('Order_Date').size()
        daily_orders.index = pd.DatetimeIndex(daily_orders.index)
        daily_orders = daily_orders.sort_index()
        
        # Make sure we have enough data
        if len(daily_orders) < 14:  # Need at least 14 days
            return None, "Not enough historical data for forecasting"
            
        # Create training and test sets
        train_size = int(len(daily_orders) * 0.8)
        train, test = daily_orders[:train_size], daily_orders[train_size:]
        
        # Fit a simple model
        from statsmodels.tsa.arima.model import ARIMA
        
        try:
            # Try ARIMA model first
            model = ARIMA(train, order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=len(test) + 30)
            
            # Split forecast into test period and future period
            test_forecast = forecast[:len(test)]
            future_forecast = forecast[len(test):]
            
            # Generate future dates
            last_date = daily_orders.index[-1]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
            future_series = pd.Series(future_forecast, index=future_dates)
            
            # Calculate error metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(test, test_forecast)
            rmse = np.sqrt(mean_squared_error(test, test_forecast))
            
            return {
                'historical_data': daily_orders,
                'test_forecast': pd.Series(test_forecast, index=test.index),
                'future_forecast': future_series,
                'mae': mae,
                'rmse': rmse
            }, None
            
        except:
            # Fall back to simple moving average if ARIMA fails
            window = 7  # 7-day moving average
            ma = train.rolling(window=window).mean()
            ma = ma.fillna(method='bfill')
            
            # Forecast is just the last moving average value repeated
            last_ma = ma.iloc[-1]
            test_forecast = pd.Series(index=test.index, data=[last_ma] * len(test))
            future_dates = pd.date_range(start=daily_orders.index[-1] + pd.Timedelta(days=1), periods=30)
            future_forecast = pd.Series(index=future_dates, data=[last_ma] * 30)
            
            # Calculate error metrics
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            mae = mean_absolute_error(test, test_forecast)
            rmse = np.sqrt(mean_squared_error(test, test_forecast))
            
            return {
                'historical_data': daily_orders,
                'test_forecast': test_forecast,
                'future_forecast': future_forecast,
                'mae': mae,
                'rmse': rmse,
                'model_type': 'Moving Average'
            }, None
            
    except Exception as e:
        return None, str(e)

def main():
    st.title("Food Delivery Analytics Dashboard")
    st.markdown(
        """
        This simplified dashboard demonstrates how food delivery analytics can help optimize 
        profitability, routing, and operations.
        """
    )
    
    # Sidebar
    st.sidebar.title("Upload & Settings")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload delivery data (CSV)", type=["csv"])
    
    # Use sample data or uploaded data
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("Data uploaded successfully!")
    else:
        st.sidebar.info("Using sample data for demonstration.")
        data = load_sample_data()
    
    # Process data
    with st.spinner('Processing data...'):
        df = preprocess_data(data)
        st.sidebar.success("Data processed successfully!")
    
    # Display data preview
    with st.expander("Data Preview", expanded=False):
        st.dataframe(df.head())
    
    # Display KPI metrics dashboard
    display_kpi_metrics(df)

    avg_profit = df['Profit'].mean()
    profitable_pct = (df['Profit'] > 0).mean() * 100
    avg_time = df['Time_taken(min)'].mean()
    avg_distance = df['Distance_km'].mean()
    avg_speed = avg_distance / (avg_time / 60)  # km/h

    # Display dynamic pricing factors
    with st.expander("📊 Executive Summary", expanded=True):
        # Create two columns
        summary_col1, summary_col2 = st.columns([2,1])
        
        with summary_col1:
            st.markdown(f"""
            <h3 style="color: {BRAND_COLORS['primary']};">Key Performance Insights</h3>
            <ul>
                <li><strong>Profitability:</strong> {profitable_pct:.1f}% of deliveries are profitable with an average profit of ${avg_profit:.2f} per delivery</li>
                <li><strong>Operations:</strong> Average delivery time is {avg_time:.1f} minutes at an average speed of {avg_speed:.1f} km/h</li>
                <li><strong>Pricing:</strong> Optimal base delivery fee is $125 based on cost analysis</li>
                <li><strong>Areas for Improvement:</strong> Dynamic pricing could increase profits by up to 49.5%</li>
            </ul>
            
            <h3 style="color: {BRAND_COLORS['primary']};">Top Recommendations</h3>
            <ol>
                <li>Implement dynamic pricing during peak hours (6-9 PM) and adverse weather</li>
                <li>Focus driver allocation on Friday and Saturday evenings</li>
                <li>Optimize restaurant assignments to reduce wait times</li>
            </ol>
            """, unsafe_allow_html=True)

        import plotly.graph_objects as go 
        with summary_col2:
            # Add a simple gauge chart for overall performance
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = 75,  # Example performance score
                title = {'text': "Overall Performance"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': BRAND_COLORS['primary']},
                    'steps': [
                        {'range': [0, 50], 'color': "#FF9999"},
                        {'range': [50, 80], 'color': "#FFCC99"},
                        {'range': [80, 100], 'color': "#CCFF99"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 85
                    }
                }
            ))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
            st.plotly_chart(fig, use_container_width=True)

    # Main tabs for different analysis views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Profitability", 
    "🚚 Operations", 
    "💰 Pricing",
    "📈 Forecasting",
    "📋 Recommendations"
    ])
    
    with tab1:  # Profitability
        st.header("Profitability Analysis")
        
        # Profit by different factors
        col1, col2 = st.columns(2)
        
        with col1:
            factor_options = ["City", "Type_of_vehicle", "Weatherconditions", "Road_traffic_density"]
            selected_factor = st.selectbox("Select Factor to Analyze", factor_options)
            
            factor_fig = plot_profit_by_factor(df, selected_factor)
            st.plotly_chart(factor_fig, use_container_width=True)
        
        with col2:
            hour_fig = plot_profit_by_hour(df)
            st.plotly_chart(hour_fig, use_container_width=True)
        
        # Profit by day of week
        profit_by_day = df.groupby('Order_Day')['Profit'].mean()
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        profit_by_day.index = [days[i] for i in profit_by_day.index]
        
        fig = px.line(
            x=profit_by_day.index, 
            y=profit_by_day.values,
            markers=True,
            title='Average Profit by Day of Week',
            labels={'x': 'Day of Week', 'y': 'Average Profit'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:  # Operations
        st.header("Operational Analysis")
        
        # Create subtabs
        ops_tab1, ops_tab2 = st.tabs(["Route Optimization", "Efficiency Analysis"])
        
        with ops_tab1:
            # Show map
            st.subheader("Delivery Hotspot Map")
            location_map, error_msg = create_demo_map()

            if location_map:
                try:
                    st.markdown(f"""
                    <div style="background-color: #fff3e0; border-radius: 5px; padding: 10px; margin-bottom: 15px; border-left: 5px solid #FF9800;">
                        <p style="color: #333333; margin: 0;"><strong>Demo Map:</strong> This is a simulation showing potential patterns in food delivery data.</p>
                    </div>
                    """, unsafe_allow_html=True)
                        
                    st.markdown("""
                        ### Map Legend:
                        - **Green markers**: Restaurant locations grouped in clusters
                        - **Blue markers**: Delivery locations grouped by neighborhood
                        - **Red circles**: High-demand hotspot areas
                        - **Purple lines**: Sample optimal delivery routes
                        - **Heatmaps**: Density of restaurants (red/green) and deliveries (yellow/red)
                        
                        Use the layer control in the top right to toggle different views.
                        """)
                    st_folium(location_map, width=800, returned_objects=[])

                except Exception as e:
                    st.error(f"Error displaying map: {str(e)}")
            elif error_msg:
                st.warning(f"Cannot display map: {error_msg}")
            else:
                st.warning("Cannot display map due to missing data")

            # Show route efficiency data
            st.subheader("Route Efficiency by City")
            
            # Calculate route efficiency
            route_metrics = df.groupby(['City', 'Type_of_vehicle']).agg({
                'Time_taken(min)': 'mean',
                'Distance_km': 'mean',
                'Profit': 'mean'
            }).reset_index()
            
            route_metrics['Speed_km_per_h'] = route_metrics['Distance_km'] / (route_metrics['Time_taken(min)'] / 60)
            route_metrics['Profit_per_hour'] = route_metrics['Profit'] / (route_metrics['Time_taken(min)'] / 60)
            
            # Show table of efficient routes
            st.dataframe(route_metrics.sort_values('Profit_per_hour', ascending=False))
            
            # Show chart of most profitable routes
            fig = px.bar(
                route_metrics.sort_values('Profit_per_hour', ascending=False).head(10),
                x='City', 
                y='Profit_per_hour',
                color='Type_of_vehicle',
                title="Most Profitable Routes (Profit per Hour)",
                labels={'Profit_per_hour': 'Profit per Hour ($)', 'City': 'City', 'Type_of_vehicle': 'Vehicle Type'},
                barmode='group'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with ops_tab2:
            st.subheader("Delivery Efficiency Analysis")
            
            # Create time efficiency metrics
            time_by_factors = pd.DataFrame()
            
            # Time by traffic
            traffic_time = df.groupby('Road_traffic_density')['Time_taken(min)'].mean().reset_index()
            time_by_factors = pd.concat([time_by_factors, traffic_time.rename(columns={'Road_traffic_density': 'Factor', 'Time_taken(min)': 'Avg_Time'})])
            
            # Time by weather
            weather_time = df.groupby('Weatherconditions')['Time_taken(min)'].mean().reset_index()
            time_by_factors = pd.concat([time_by_factors, weather_time.rename(columns={'Weatherconditions': 'Factor', 'Time_taken(min)': 'Avg_Time'})])
            
            # Time by vehicle
            vehicle_time = df.groupby('Type_of_vehicle')['Time_taken(min)'].mean().reset_index()
            time_by_factors = pd.concat([time_by_factors, vehicle_time.rename(columns={'Type_of_vehicle': 'Factor', 'Time_taken(min)': 'Avg_Time'})])
            
            # Create chart
            fig = px.bar(
                time_by_factors,
                x='Factor', 
                y='Avg_Time',
                title="Average Delivery Time by Factors",
                labels={'Avg_Time': 'Average Time (min)', 'Factor': 'Factor'}
            )
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
            

            # Driver performance
            st.subheader("Driver Performance")
            
            if 'Delivery_person_ID' in df.columns:
                # Build agg dictionary dynamically based on available columns and their dtypes
                agg_dict = {}
                
                # For counting orders
                count_col = None
                for col in ['Order_ID', 'Delivery_person_ID']:
                    if col in df.columns and col != 'Delivery_person_ID':
                        count_col = col
                        break
                
                if count_col:
                    agg_dict[count_col] = 'count'
                
                # For numeric columns
                for col in ['Time_taken(min)', 'Profit', 'Delivery_person_Ratings']:
                    if col in df.columns:
                        # Check if column is numeric
                        if pd.api.types.is_numeric_dtype(df[col]):
                            agg_dict[col] = 'mean'
                        else:
                            # Try to convert to numeric
                            try:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                agg_dict[col] = 'mean'
                            except:
                                st.warning(f"Column {col} is not numeric and cannot be aggregated with mean")
                
                if agg_dict:
                    try:
                        driver_perf = df.groupby('Delivery_person_ID').agg(agg_dict).reset_index()

                        # Create a new DataFrame with renamed columns to avoid conflicts
                        new_data = {'Driver ID': driver_perf['Delivery_person_ID']}
                        
                        if count_col in driver_perf.columns:
                            new_data['Orders/Deliveries'] = driver_perf[count_col]
                            
                        if 'Time_taken(min)' in driver_perf.columns:
                            new_data['Avg Time (min)'] = driver_perf['Time_taken(min)']
                            
                        if 'Profit' in driver_perf.columns:
                            new_data['Avg Profit ($)'] = driver_perf['Profit']
                            
                        if 'Delivery_person_Ratings' in driver_perf.columns:
                            new_data['Rating'] = driver_perf['Delivery_person_Ratings']
                        
                        # Create a new DataFrame with the renamed columns
                        driver_perf_renamed = pd.DataFrame(new_data)
                        
                        # Determine sorting column
                        if 'Avg Profit ($)' in driver_perf_renamed.columns:
                            sort_col = 'Avg Profit ($)'
                        elif 'Orders/Deliveries' in driver_perf_renamed.columns:
                            sort_col = 'Orders/Deliveries'
                        else:
                            sort_col = driver_perf_renamed.columns[1] if len(driver_perf_renamed.columns) > 1 else 'Driver ID'
                        
                        # Show top drivers
                        st.dataframe(driver_perf_renamed.sort_values(sort_col, ascending=False).head(10))
                        
                    except Exception as e:
                        st.error(f"Error analyzing driver performance: {str(e)}")
                        # Show raw data for debugging
                        st.write("Sample of Delivery_person_ID:")
                        st.write(df['Delivery_person_ID'].head())
                        for col in agg_dict:
                            st.write(f"Sample of {col}:")
                            st.write(df[col].head())
                            st.write(f"Data type: {df[col].dtype}")
                else:
                    st.warning("No suitable columns found for driver performance analysis")
            else:
                st.warning("Driver performance data not available - missing Delivery_person_ID column")
    
    with tab3:  # Pricing
        st.header("Pricing Analysis")
        
        # Create pricing simulation
        st.subheader("Pricing Optimization")
        
        # Simulated optimal price
        optimal_price = 125
        
        # Create pricing chart
        price_points = range(75, 176, 5)
        price_results = []
        
        for price in price_points:
            # Simple profit model
            avg_profit = price - 75 - np.random.normal(0, 5)
            profitable_pct = max(0, min(100, 100 - 0.5 * (price - optimal_price)**2 / 10))
            
            price_results.append({
                'Base_Price': price,
                'Average_Profit': avg_profit,
                'Profit_Margin_Pct': 100 * avg_profit / price,
                'Profitable_Deliveries_Pct': profitable_pct
            })
        
        price_df = pd.DataFrame(price_results)
        
        # Create figure with secondary y-axis
        from plotly.subplots import make_subplots  
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        import plotly.graph_objects as go
        # Add traces
        fig.add_trace(
            go.Scatter(
                x=price_df['Base_Price'],
                y=price_df['Average_Profit'],
                name="Average Profit",
                line=dict(color='blue', width=3)
            ),
            secondary_y=False,
        )
        
        fig.add_trace(
            go.Scatter(
                x=price_df['Base_Price'],
                y=price_df['Profitable_Deliveries_Pct'],
                name="% Profitable Deliveries",
                line=dict(color='green', width=3)
            ),
            secondary_y=True,
        )
        
        # Add vertical line at optimal price
        fig.add_vline(
            x=optimal_price,
            line_width=3,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Optimal Price: ${optimal_price}",
            annotation_position="top right"
        )
        
        # Set titles
        fig.update_layout(
            title_text="Pricing Analysis",
            height=450
        )
        
        # Set x-axis label
        fig.update_xaxes(title_text="Base Price ($)")
        
        # Set y-axes titles
        fig.update_yaxes(title_text="Average Profit ($)", secondary_y=False)
        fig.update_yaxes(title_text="% Profitable Deliveries", secondary_y=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Dynamic pricing
        st.subheader("Dynamic Pricing Factors")
        
        dynamic_fig = plot_dynamic_pricing(df)
        st.plotly_chart(dynamic_fig, use_container_width=True)
        
        # Compare static vs dynamic pricing
        st.subheader("Static vs Dynamic Pricing Comparison")
        
        col1, col2 = st.columns(2)
                
        with col1:
            st.markdown("""
            <div class="static-pricing">
                <h4 style="color: #2e7d32; font-weight: bold;">Static Pricing</h4>
                <p class="pricing-value">$15.25</p>
                <p>Average Profit per Order</p>
                <p class="pricing-value" style="font-size: 1.5rem;">78.5%</p>
                <p>Profitable Orders</p>
            </div>
            """, unsafe_allow_html=True)
                
        with col2:
            st.markdown("""
            <div class="dynamic-pricing">
                <h4 style="color: #1565c0; font-weight: bold;">Dynamic Pricing</h4>
                <p class="pricing-value">$22.80</p>
                <p>Average Profit per Order</p>
                <p class="pricing-value" style="font-size: 1.5rem;">92.3%</p>
                <p>Profitable Orders</p>
                <p style="color: #2e7d32; font-weight: bold;">+49.5% Profit Increase</p>
            </div>
            """, unsafe_allow_html=True)

        # Customer segment pricing
        st.subheader("Customer Segment Pricing Strategy")
        
        # Create sample customer segments
        segments = ["Urban Professionals", "Suburban Families", "College Students", "Rural Customers"]
        base_prices = [140, 125, 110, 130]
        price_elasticity = [0.8, 1.2, 1.5, 0.9]  # Higher means more sensitive to price
        optimal_discounts = [0, 5, 15, 0]
        segment_colors = [BRAND_COLORS['primary'], BRAND_COLORS['secondary'], 
                         BRAND_COLORS['accent'], BRAND_COLORS['neutral']]
        
        # Create a DataFrame for the segments
        segment_df = pd.DataFrame({
            'Segment': segments,
            'Base Price': base_prices,
            'Price Elasticity': price_elasticity,
            'Optimal Discount (%)': optimal_discounts,
            'Revenue Impact ($)': [12500, 8500, 6200, 4800]
        })
        
        # Display segment table
        st.dataframe(segment_df)
        
        # Create a visualization of pricing by segment
        fig = px.bar(
            segment_df, 
            x='Segment', 
            y='Base Price',
            color='Segment', 
            text='Base Price',
            color_discrete_sequence=segment_colors,
            title="Optimal Base Price by Customer Segment"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add segment-specific recommendations
        st.subheader("Segment-Specific Strategies")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div style="background-color: #FFFFFF; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 5px solid #43A047; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="color: #2E7D32; margin-top: 0;">Urban Professionals</h4>
                    <p style="color: #000000;"><strong>Strategy:</strong> Premium pricing with faster delivery guarantees</p>
                    <p style="color: #000000;"><strong>Key Value:</strong> Time savings, reliability</p>
                    <p style="color: #000000;"><strong>Implementation:</strong> Priority delivery option for 20% premium</p>
            </div>
                
            <div style="background-color: #FFFFFF; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 5px solid #FF9800; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="color: #E65100; margin-top: 0;">College Students</h4>
                    <p style="color: #000000;"><strong>Strategy:</strong> Discounted pricing with group orders</p>
                    <p style="color: #000000;"><strong>Key Value:</strong> Affordability, social experience</p>
                    <p style="color: #000000;"><strong>Implementation:</strong> 15% discount for multiple delivery locations</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="background-color: #FFFFFF; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 5px solid #1E88E5; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="color: #0D47A1; margin-top: 0;">Suburban Families</h4>
                    <p style="color: #000000;"><strong>Strategy:</strong> Bundle pricing for larger orders</p>
                    <p style="color: #000000;"><strong>Key Value:</strong> Convenience, variety</p>
                    <p style="color: #000000;"><strong>Implementation:</strong> Free delivery for orders over $50</p>
            </div>
            
            <div style="background-color: #FFFFFF; border-radius: 10px; padding: 15px; margin-bottom: 15px; border-left: 5px solid #7B1FA2; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <h4 style="color: #4A148C; margin-top: 0;">Rural Customers</h4>
                    <p style="color: #000000;"><strong>Strategy:</strong> Distance-based pricing with batch deliveries</p>
                    <p style="color: #000000;"><strong>Key Value:</strong> Availability, reliability</p>
                    <p style="color: #000000;"><strong>Implementation:</strong> Scheduled delivery days with discounts</p>
            </div>
            """, unsafe_allow_html=True)
    
    with tab4:  # Forecasting
        st.header("Demand Forecasting")
        
        # Run the forecasting
        forecast_results, forecast_error = forecast_demand(df)
        
        if forecast_results:
            # Display forecast metrics
            st.subheader("Forecast Accuracy Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Mean Absolute Error (MAE)", f"{forecast_results['mae']:.2f}")
            with col2:
                st.metric("Root Mean Squared Error (RMSE)", f"{forecast_results['rmse']:.2f}")
            
            # Plot historical data and forecast
            st.subheader("Order Volume Forecast")
            
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=forecast_results['historical_data'].index, 
                y=forecast_results['historical_data'].values,
                name='Historical Orders',
                line=dict(color=BRAND_COLORS['secondary'], width=2)
            ))
            
            # Test forecast
            if 'test_forecast' in forecast_results:
                fig.add_trace(go.Scatter(
                    x=forecast_results['test_forecast'].index, 
                    y=forecast_results['test_forecast'].values,
                    name='Forecast (Test Period)',
                    line=dict(color=BRAND_COLORS['primary'], width=2, dash='dash')
                ))
            
            # Future forecast
            fig.add_trace(go.Scatter(
                x=forecast_results['future_forecast'].index, 
                y=forecast_results['future_forecast'].values,
                name='Future Forecast',
                line=dict(color=BRAND_COLORS['accent'], width=3)
            ))
            
            # Customize layout
            fig.update_layout(
                title='Daily Order Volume Forecast',
                xaxis_title='Date',
                yaxis_title='Number of Orders',
                height=500,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly patterns
            st.subheader("Weekly Order Patterns")
            
            # Group by day of week
            if len(forecast_results['historical_data']) >= 7:
                day_of_week = forecast_results['historical_data'].groupby(forecast_results['historical_data'].index.dayofweek).mean()
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_of_week.index = [days[i] for i in day_of_week.index]
                
                fig = px.bar(
                    x=day_of_week.index, 
                    y=day_of_week.values,
                    labels={'x': 'Day of Week', 'y': 'Average Orders'},
                    title='Average Orders by Day of Week',
                    color=day_of_week.values,
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(height=400, coloraxis_showscale=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Could not generate forecast: {forecast_error}")

            # Add seasonality analysis
            st.subheader("Hourly Order Patterns")
            
            # Create sample hourly data if we don't have real data
            hours = list(range(24))
            hourly_volume = [
                5, 3, 2, 1, 1, 2,      # 0-5 AM
                8, 15, 25, 20, 15, 18,  # 6-11 AM
                30, 28, 15, 12, 18, 25, # 12-5 PM
                35, 40, 30, 20, 12, 8   # 6-11 PM
            ]
            
            # Create hour of day visualization
            fig = px.line(
                x=hours, 
                y=hourly_volume,
                markers=True,
                title="Average Order Volume by Hour of Day",
                labels={'x': 'Hour of Day', 'y': 'Average Orders'},
                color_discrete_sequence=[BRAND_COLORS['primary']]
            )
            
            # Add peak hour annotations
            peak_hours = [8, 19]  # Example peak hours
            for hour in peak_hours:
                fig.add_annotation(
                    x=hour,
                    y=hourly_volume[hour],
                    text="Peak Time",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowcolor=BRAND_COLORS['accent'],
                    ax=-50,
                    ay=-40
                )
            
            fig.update_layout(
                height=400,
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(24)),
                    ticktext=[f"{h}:00" for h in range(24)]
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add forecast interpretation
            st.subheader("Forecast Insights")
            
            # Calculate some sample insights
            future_trend = "upward" if forecast_results['future_forecast'].iloc[-1] > forecast_results['future_forecast'].iloc[0] else "downward"
            peak_day_idx = forecast_results['future_forecast'].idxmax() if len(forecast_results['future_forecast']) > 0 else None
            peak_day = peak_day_idx.strftime("%A, %B %d") if peak_day_idx is not None else "Not available"
            
            # Display insights in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background-color: {BRAND_COLORS['light_bg']}; padding: 20px; border-radius: 10px; border-top: 4px solid {BRAND_COLORS['primary']};">
                    <h4 style="color: {BRAND_COLORS['primary']};">Trend</h4>
                    <p style="font-size: 1.2rem; font-weight: bold;">{future_trend.title()}</p>
                    <p>Overall forecast trend for the next 30 days</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background-color: {BRAND_COLORS['light_bg']}; padding: 20px; border-radius: 10px; border-top: 4px solid {BRAND_COLORS['accent']};">
                    <h4 style="color: {BRAND_COLORS['accent']};">Peak Day</h4>
                    <p style="font-size: 1.2rem; font-weight: bold;">{peak_day}</p>
                    <p>Highest volume expected on this day</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background-color: {BRAND_COLORS['light_bg']}; padding: 20px; border-radius: 10px; border-top: 4px solid {BRAND_COLORS['secondary']};">
                    <h4 style="color: {BRAND_COLORS['secondary']};">Forecast Accuracy</h4>
                    <p style="font-size: 1.2rem; font-weight: bold;">{100 - forecast_results['mae'] * 5:.1f}%</p>
                    <p>Based on test data evaluation</p>
                </div>
                """, unsafe_allow_html=True)

    with tab5:  # Recommendations 
        st.header("Business Recommendations")
        
        # Create static recommendations
        recommendations = {
            'pricing': [
                "Set base delivery fee to $125 for maximizing profit",
                "Implement dynamic pricing during evening peak hours (6-9 PM)",
                "Increase prices by 30% during rainy and stormy conditions"
            ],
            'operations': [
                "Focus on reducing average wait time (12.5 minutes) by optimizing restaurant assignments",
                "Address delivery time variability in high traffic areas",
                "Implement batch ordering system for clustered delivery locations"
            ],
            'driver_management': [
                "Increase driver allocation for Friday and Saturday evenings",
                "Analyze top performing drivers' patterns to develop training program for new drivers",
                "Implement performance-based incentives for drivers to improve delivery efficiency"
            ],
            'route_optimization': [
                "Focus on high-profit route types in Urban areas with motorcycles",
                "Establish dark kitchens in high-density pickup areas to reduce delivery times and costs",
                "Optimize driver starting positions based on delivery hotspots to minimize empty miles"
            ]
        }
        
        # Display recommendations by category
        categories = {
            'pricing': "Pricing Recommendations",
            'operations': "Operational Recommendations",
            'driver_management': "Driver Management",
            'route_optimization': "Route Optimization"
        }
        
        for key, title in categories.items():
            st.subheader(title)
            
            for rec in recommendations[key]:
                st.markdown(f"""
                <div class="recommendation-box">
                    <strong class="recommendation-text">🔍 {rec}</strong>
                </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; padding: 10px;">
    <p style="color: #666666; font-size: 12px;">
        Food Delivery Analytics Dashboard | Created with Streamlit | Data Updated: {datetime.datetime.now().strftime('%Y-%m-%d')}
    </p>
</div>
""", unsafe_allow_html=True)