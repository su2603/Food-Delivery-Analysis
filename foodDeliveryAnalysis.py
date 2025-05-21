import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import radians, cos, sin, asin, sqrt
import os
import logging
import joblib
import warnings
from datetime import datetime, timedelta
import requests
from geopy.distance import geodesic
import folium
from folium.plugins import HeatMap, MarkerCluster
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA

# For modeling
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.impute import KNNImputer

# For advanced modeling
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# For explainability
import shap

# For parallel processing
from joblib import Parallel, delayed
import multiprocessing

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("food_delivery_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants for the analysis
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

class FoodDeliveryAnalysis:
    """
    A comprehensive class for analyzing and optimizing 
    food delivery operations and profitability.
    """
    
    def __init__(self, data_path=None, df=None):
        """
        Initialize the FoodDeliveryAnalysis object.
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the CSV file containing the food delivery data
        df : pandas.DataFrame, optional
            DataFrame containing the food delivery data
        """
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.results = {}
        self.visualizations = {}
        self.clusters = {}
        
        # Business model parameters (can be tuned)
        self.business_params = {
            'base_delivery_fee': 100,  # Base delivery fee in currency units
            'cost_per_min': 2,         # Cost per minute of delivery time
            'cost_per_km': 3,          # Cost per kilometer traveled
            'fixed_costs': 20,         # Fixed costs per delivery
            'driver_base_pay': 30,     # Base pay for driver
            'fuel_cost_per_km': 1.5,   # Fuel cost per kilometer
            'vehicle_maintenance': 0.5 # Vehicle maintenance cost per kilometer
        }
        
        # Environmental parameters
        self.env_params = {
            'co2_emissions_per_km': 0.12,  # CO2 emissions per km in kg
            'fuel_efficiency': 15         # km per liter
        }
        
        if data_path is not None:
            logger.info(f"Loading data from {data_path}")
            self.df = pd.read_csv(data_path)
        elif df is not None:
            logger.info("Using provided DataFrame")
            self.df = df.copy()
        else:
            logger.warning("No data provided. You'll need to load data later.")
            self.df = None
    
    def load_data(self, data_path):
        """Load data from a CSV file."""
        logger.info(f"Loading data from {data_path}")
        try:
            self.df = pd.read_csv(data_path)
            logger.info(f"Data loaded successfully with {self.df.shape[0]} rows and {self.df.shape[1]} columns.")
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
    def preprocess_data(self):
        """Preprocess the data for analysis."""
        if self.df is None:
            logger.error("No data available. Load data first.")
            return False
        
        logger.info("Starting data preprocessing")
        try:
            # Make a copy to avoid modifying the original
            df = self.df.copy()
            
            # --- Basic Cleaning ---
            # Extract time taken (removing 'min' suffix and converting to numeric)
            df['Time_taken(min)'] = df['Time_taken(min)'].str.extract('(\d+)').astype(float)
            
            # Convert age and rating to numeric
            df['Delivery_person_Age'] = pd.to_numeric(df['Delivery_person_Age'], errors='coerce')
            df['Delivery_person_Ratings'] = pd.to_numeric(df['Delivery_person_Ratings'], errors='coerce')
            
            # --- Advanced Geospatial Processing ---
            # Calculate distance using Haversine formula
            logger.info("Calculating distances and geospatial features")
            df['Distance_km'] = df.apply(self._calculate_haversine_distance, axis=1)
            
            # Create straight-line vs actual path ratio (approximate)
            # We'll use a multiplier to simulate real-world routes which are rarely straight lines
            df['Route_Efficiency'] = np.random.uniform(1.0, 1.5, size=len(df))
            df['Actual_Distance_km'] = df['Distance_km'] * df['Route_Efficiency']
            
            # --- DateTime Processing ---
            logger.info("Processing datetime features")
            # Parse date
            df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%d-%m-%Y', errors='coerce')
            
            # Handle missing time values before datetime conversion
            df['Time_Orderd'] = df['Time_Orderd'].fillna('00:00')
            df['Time_Order_picked'] = df['Time_Order_picked'].fillna('00:00')
            
            # Create datetime columns for order and pickup times
            df['Order_Datetime'] = pd.to_datetime(
                df['Order_Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time_Orderd'], 
                format='%Y-%m-%d %H:%M', 
                errors='coerce'
            )
            df['Pickup_Datetime'] = pd.to_datetime(
                df['Order_Date'].dt.strftime('%Y-%m-%d') + ' ' + df['Time_Order_picked'], 
                format='%Y-%m-%d %H:%M', 
                errors='coerce'
            )
            
            # Extract features from datetime
            df['Order_Hour'] = df['Order_Datetime'].dt.hour
            df['Pickup_Hour'] = df['Pickup_Datetime'].dt.hour
            df['Order_Minute'] = df['Order_Datetime'].dt.minute
            df['Pickup_Minute'] = df['Pickup_Datetime'].dt.minute
            df['DayOfWeek'] = df['Order_Date'].dt.dayofweek
            df['Month'] = df['Order_Date'].dt.month
            df['Year'] = df['Order_Date'].dt.year
            df['Day'] = df['Order_Date'].dt.day
            df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)  # 5,6 = Saturday, Sunday
            df['IsHoliday'] = self._is_holiday(df['Order_Date']).astype(int)
            df['IsPeakHour'] = ((df['Order_Hour'] >= 11) & (df['Order_Hour'] <= 14) | 
                               (df['Order_Hour'] >= 18) & (df['Order_Hour'] <= 21)).astype(int)
            
            # Calculate wait time (time between order and pickup) in minutes
            df['Wait_Time_Minutes'] = ((df['Pickup_Datetime'] - df['Order_Datetime']).dt.total_seconds() / 60)
            # Handle negative or extreme wait times
            df['Wait_Time_Minutes'] = np.where(
                (df['Wait_Time_Minutes'] < 0) | (df['Wait_Time_Minutes'] > 120),
                np.nan,
                df['Wait_Time_Minutes']
            )
            
            # Impute missing wait times using KNN
            wait_time_imputer = KNNImputer(n_neighbors=5)
            df['Wait_Time_Minutes'] = wait_time_imputer.fit_transform(
                df
            )
            
            # --- Advanced Feature Engineering ---
            logger.info("Creating advanced features")
            
            # Delivery speed (km per minute)
            df['Delivery_Speed'] = df['Distance_km'] / df['Time_taken(min)']
            
            # Restaurant preparation time (estimated from wait time)
            df['Restaurant_Prep_Time'] = df['Wait_Time_Minutes'] * 0.7  # Assuming 70% of wait time is prep
            
            # Delivery person efficiency (compare to average delivery speed)
            avg_speed = df.groupby('Delivery_person_ID')['Delivery_Speed'].transform('mean')
            df['Delivery_Person_Efficiency'] = df['Delivery_Speed'] / avg_speed
            
            # Restaurant efficiency (compare to average prep time)
            avg_prep = df.groupby('Restaurant_ID')['Restaurant_Prep_Time'].transform('mean')
            df['Restaurant_Efficiency'] = avg_prep / df['Restaurant_Prep_Time']
            
            # --- Calculate Costs and Profits ---
            logger.info("Calculating costs and profits")
            
            # Calculate variable costs
            df['Time_Cost'] = df['Time_taken(min)'] * self.business_params['cost_per_min']
            df['Distance_Cost'] = df['Distance_km'] * self.business_params['cost_per_km']
            df['Fuel_Cost'] = df['Distance_km'] * self.business_params['fuel_cost_per_km']
            df['Maintenance_Cost'] = df['Distance_km'] * self.business_params['vehicle_maintenance']
            
            # Calculate total cost
            df['Total_Variable_Cost'] = (df['Time_Cost'] + df['Distance_Cost'] + 
                                       df['Fuel_Cost'] + df['Maintenance_Cost'])
            df['Total_Cost'] = df['Total_Variable_Cost'] + self.business_params['fixed_costs'] + self.business_params['driver_base_pay']
            
            # Calculate profit
            df['Revenue'] = self.business_params['base_delivery_fee']
            df['Profit'] = df['Revenue'] - df['Total_Cost']
            
            # Calculate environmental impact
            df['CO2_Emissions'] = df['Distance_km'] * self.env_params['co2_emissions_per_km']
            df['Fuel_Consumption'] = df['Distance_km'] / self.env_params['fuel_efficiency']
            
            # --- Location Clustering ---
            logger.info("Performing location clustering")
            # Get location data
            location_cols = ['Restaurant_latitude', 'Restaurant_longitude', 
                             'Delivery_location_latitude', 'Delivery_location_longitude']
            location_df = df[location_cols].dropna()
            
            # Cluster restaurant locations
            if len(location_df) > 0:
                # Restaurant clusters
                restaurant_coords = location_df.values
                restaurant_clustering = DBSCAN(eps=0.02, min_samples=5).fit(restaurant_coords)
                restaurant_labels = restaurant_clustering.labels_
                
                # Handle outliers (DBSCAN marks them as -1)
                num_outliers = np.sum(restaurant_labels == -1)
                if num_outliers > 0:
                    logger.info(f"Found {num_outliers} restaurant location outliers")
                    # Assign outliers to nearest cluster or create new clusters
                    kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE)
                    outlier_indices = np.where(restaurant_labels == -1)[0]
                    outlier_coords = restaurant_coords[outlier_indices]
                    outlier_labels = kmeans.fit_predict(outlier_coords)
                    # Assign new cluster IDs starting from max existing ID + 1
                    if len(np.unique(restaurant_labels)) > 1:
                        max_label = np.max(restaurant_labels[restaurant_labels != -1])
                        restaurant_labels[outlier_indices] = outlier_labels + max_label + 1
                    else:
                        restaurant_labels[outlier_indices] = outlier_labels
                
                # Create a mapping from index to cluster label
                restaurant_cluster_map = dict(zip(location_df.index, restaurant_labels))
                # Apply cluster labels to original dataframe
                df['Restaurant_Cluster'] = df.index.map(restaurant_cluster_map).fillna(-1).astype(int)
                
                # Delivery clusters (same process)
                delivery_coords = location_df.values
                delivery_clustering = DBSCAN(eps=0.02, min_samples=5).fit(delivery_coords)
                delivery_labels = delivery_clustering.labels_
                
                # Handle outliers
                num_outliers = np.sum(delivery_labels == -1)
                if num_outliers > 0:
                    logger.info(f"Found {num_outliers} delivery location outliers")
                    kmeans = KMeans(n_clusters=5, random_state=RANDOM_STATE)
                    outlier_indices = np.where(delivery_labels == -1)[0]
                    outlier_coords = delivery_coords[outlier_indices]
                    outlier_labels = kmeans.fit_predict(outlier_coords)
                    if len(np.unique(delivery_labels)) > 1:
                        max_label = np.max(delivery_labels[delivery_labels != -1])
                        delivery_labels[outlier_indices] = outlier_labels + max_label + 1
                    else:
                        delivery_labels[outlier_indices] = outlier_labels
                
                delivery_cluster_map = dict(zip(location_df.index, delivery_labels))
                df['Delivery_Cluster'] = df.index.map(delivery_cluster_map).fillna(-1).astype(int)
                
                # Create route type feature
                df['Route_Type'] = df['Restaurant_Cluster'].astype(str) + '_' + df['Delivery_Cluster'].astype(str)
                
                # Save cluster models
                self.clusters['restaurant_model'] = restaurant_clustering
                self.clusters['delivery_model'] = delivery_clustering
            
            # --- Customer Segmentation ---
            logger.info("Performing customer segmentation")
            # Extract customer features
            customer_features = ['Type_of_order', 'Type_of_vehicle', 'City', 'Distance_km']
            # If the customer ID is available, we could also add historical data
            
            # Create customer segments using K-means
            if 'Customer_ID' in df.columns:
                customer_df = df.groupby('Customer_ID').agg({
                    'Type_of_order': lambda x: x.value_counts().index[0],
                    'Type_of_vehicle': lambda x: x.value_counts().index[0],
                    'City': lambda x: x.value_counts().index[0],
                    'Distance_km': 'mean',
                    'Time_taken(min)': 'mean',
                    'Profit': 'mean',
                    'Order_ID': 'count'
                }).reset_index()
                
                customer_df.rename(columns={'Order_ID': 'Order_Count'}, inplace=True)
                
                # Prepare data for clustering
                cat_features = ['Type_of_order', 'Type_of_vehicle', 'City']
                num_features = ['Distance_km', 'Time_taken(min)', 'Profit', 'Order_Count']
                
                # Encode categorical features
                customer_encoder = OneHotEncoder(handle_unknown='ignore')
                customer_cat_encoded = customer_encoder.fit_transform(customer_df[cat_features])
                
                # Scale numerical features
                customer_scaler = StandardScaler()
                customer_num_scaled = customer_scaler.fit_transform(customer_df[num_features])
                
                # Combine features
                customer_features_combined = np.hstack((customer_cat_encoded.toarray(), customer_num_scaled))
                
                # Cluster customers
                customer_kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE)
                customer_df['Customer_Segment'] = customer_kmeans.fit_predict(customer_features_combined)
                
                # Map customer segments back to original dataframe
                customer_segment_map = dict(zip(customer_df['Customer_ID'], customer_df['Customer_Segment']))
                df['Customer_Segment'] = df['Customer_ID'].map(customer_segment_map)
                
                # Save customer segmentation model
                self.clusters['customer_model'] = customer_kmeans
                self.encoders['customer_encoder'] = customer_encoder
                self.scalers['customer_scaler'] = customer_scaler
            
            # --- Final Dataset ---
            # Drop rows with missing values in essential columns
            essential_columns = [
                'Time_taken(min)', 'Distance_km', 'Delivery_person_Age', 
                'Delivery_person_Ratings', 'Order_Hour', 'Pickup_Hour'
            ]
            df_cleaned = df.dropna(subset=essential_columns)
            
            # Save preprocessed data
            self.df_preprocessed = df
            self.df_cleaned = df_cleaned
            
            logger.info(f"Preprocessing complete. Clean data shape: {df_cleaned.shape}")
            return True
        
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _calculate_haversine_distance(self, row):
        """Calculate haversine distance between two coordinates."""
        try:
            lat1, lon1, lat2, lon2 = map(float, [
                row['Restaurant_latitude'], row['Restaurant_longitude'],
                row['Delivery_location_latitude'], row['Delivery_location_longitude']
            ])
            
            # Convert decimal degrees to radians
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            
            # Haversine formula
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            r = 6371  # Radius of Earth in kilometers
            
            return c * r
        except:
            return np.nan
    
    def _is_holiday(self, dates):
        """
        Simple holiday detector based on common holidays.
        In a real implementation, this would use a holiday calendar API.
        """
        # Convert to datetime if not already
        if not isinstance(dates, pd.Series):
            dates = pd.Series(dates)
        
        # Initialize result array
        is_holiday = np.zeros(len(dates))
        
        # Check for common holidays (simplified)
        for i, date in enumerate(dates):
            if pd.notnull(date):
                # New Year's Day
                if date.month == 1 and date.day == 1:
                    is_holiday[i] = 1
                # Christmas
                elif date.month == 12 and date.day == 25:
                    is_holiday[i] = 1
                # Add more holidays as needed
        
        return is_holiday
    
    def analyze_profitability(self):
        """Analyze profitability of food delivery operations."""
        if not hasattr(self, 'df_cleaned'):
            logger.error("No preprocessed data available. Run preprocess_data first.")
            return False
        
        logger.info("Starting profitability analysis")
        try:
            df = self.df_cleaned.copy()
            
            # 1. Overall profitability metrics
            profit_metrics = df.describe()
            profit_percentage = (df['Profit'] > 0).mean() * 100
            
            # 2. Profitability by different factors
            profitability_factors = [
                ('City', 'City'),
                ('Type_of_vehicle', 'Vehicle Type'),
                ('Weatherconditions', 'Weather Conditions'),
                ('Road_traffic_density', 'Traffic Density'),
                ('Route_Type', 'Route Type'),
                ('IsWeekend', 'Weekend (1) vs Weekday (0)'),
                ('IsPeakHour', 'Peak Hour (1) vs Off-Peak (0)'),
                ('IsHoliday', 'Holiday (1) vs Non-Holiday (0)')
            ]
            
            profit_by_factor = {}
            for factor, label in profitability_factors:
                if factor in df.columns:
                    profit_by_factor[label] = df.groupby(factor).agg({
                        'Profit': ['mean', 'median', 'std', 'count'],
                        'Total_Cost': 'mean',
                        'Revenue': 'mean'
                    })
                    
                    # Calculate profit margin
                    profit_by_factor[label]['Profit Margin (%)'] = (
                        profit_by_factor[label][('Profit', 'mean')] / 
                        profit_by_factor[label][('Revenue', 'mean')]
                    ) * 100
                    
                    # Sort by mean profit
                    profit_by_factor[label] = profit_by_factor[label].sort_values(('Profit', 'mean'), ascending=False)
            
            # 3. Time-based profitability
            profit_by_hour = df.groupby('Order_Hour').agg({
                'Profit': ['mean', 'median', 'std', 'count'],
                'Total_Cost': 'mean',
                'Revenue': 'mean'
            })
            
            profit_by_day = df.groupby('DayOfWeek').agg({
                'Profit': ['mean', 'median', 'std', 'count'],
                'Total_Cost': 'mean',
                'Revenue': 'mean'
            })
            
            # 4. Geographical profitability
            if 'Restaurant_Cluster' in df.columns and 'Delivery_Cluster' in df.columns:
                restaurant_cluster_profit = df.groupby('Restaurant_Cluster').agg({
                    'Profit': ['mean', 'count'],
                    'Restaurant_latitude': 'mean',
                    'Restaurant_longitude': 'mean'
                })
                
                delivery_cluster_profit = df.groupby('Delivery_Cluster').agg({
                    'Profit': ['mean', 'count'],
                    'Delivery_location_latitude': 'mean',
                    'Delivery_location_longitude': 'mean'
                })
            
            # 5. Advanced analysis: Profitability score
            score_features = ['Distance_km', 'Time_taken(min)', 'Wait_Time_Minutes', 'Delivery_person_Ratings']
            factor_columns = [col for col, _ in profitability_factors if col in df.columns]
            score_df = df[score_features + factor_columns + ['Profit']].dropna()
            
            # Scale features
            scaler = MinMaxScaler()
            score_df[score_features] = scaler.fit_transform(score_df[score_features])
            
            # Create scores (higher is better)
            score_df['Distance_Score'] = 1 - score_df['Distance_km']
            score_df['Time_Score'] = 1 - score_df['Time_taken(min)']
            score_df['Wait_Score'] = 1 - score_df['Wait_Time_Minutes']
            score_df['Rating_Score'] = score_df['Delivery_person_Ratings']
            
            # Overall profit score
            score_df['Profit_Score'] = (
                score_df['Distance_Score'] + 
                score_df['Time_Score'] + 
                score_df['Wait_Score'] + 
                score_df['Rating_Score']
            ) / 4
            
            # Find the most profitable scenarios
            scenario_columns = [col for col in factor_columns if col not in ['Restaurant_Cluster', 'Delivery_Cluster']]
            
            if scenario_columns:
                profit_scenarios = score_df.groupby(scenario_columns).agg({
                    'Profit_Score': ['mean', 'std', 'count'],
                    'Profit': ['mean', 'std']
                }).sort_values(('Profit_Score', 'mean'), ascending=False)
                
                # Filter to scenarios with enough data
                profit_scenarios = profit_scenarios[profit_scenarios[('Profit_Score', 'count')] >= 10]
                
                # Store the top scenarios
                top_scenarios = profit_scenarios.head(10)
            else:
                top_scenarios = None
            
            # Store results
            self.results['profitability'] = {
                'overall_metrics': profit_metrics,
                'profit_percentage': profit_percentage,
                'profit_by_factor': profit_by_factor,
                'profit_by_hour': profit_by_hour,
                'profit_by_day': profit_by_day,
                'top_scenarios': top_scenarios
            }
            
            if 'Restaurant_Cluster' in df.columns:
                self.results['profitability']['restaurant_cluster_profit'] = restaurant_cluster_profit
                self.results['profitability']['delivery_cluster_profit'] = delivery_cluster_profit
            
            logger.info("Profitability analysis complete")
            return True
        
        except Exception as e:
            logger.error(f"Error in profitability analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def pricing_optimization(self):
        """Optimize pricing strategy for maximum profit."""
        if not hasattr(self, 'df_cleaned'):
            logger.error("No preprocessed data available. Run preprocess_data first.")
            return False
        
        logger.info("Starting pricing optimization")
        try:
            df = self.df_cleaned.copy()
            
            # Function to calculate profit at different pricing levels
            def calculate_profit_at_price(base_price, df):
                return base_price - df['Total_Cost']
            
            # Test different pricing levels
            price_points = range(50, 201, 5)  # From 50 to 200 in steps of 5
            price_results = []
            
            for price in price_points:
                profit = calculate_profit_at_price(price, df)
                avg_profit = profit.mean()
                profit_margin = avg_profit / price * 100
                profitable_pct = (profit > 0).mean() * 100
                
                price_results.append({
                    'Base_Price': price,
                    'Average_Profit': avg_profit,
                    'Profit_Margin_Pct': profit_margin,
                    'Profitable_Deliveries_Pct': profitable_pct
                })
            
            price_analysis = pd.DataFrame(price_results)
            
            # Find the optimal uniform price
            # We'll consider price points that make at least 90% of deliveries profitable
            viable_prices = price_analysis[price_analysis['Profitable_Deliveries_Pct'] >= 90]
            
            if len(viable_prices) > 0:
                uniform_optimal_price = viable_prices.sort_values('Average_Profit', ascending=False).iloc[0]
            else:
                # If no price makes 90% profitable, find the best trade-off
                uniform_optimal_price = price_analysis.sort_values('Average_Profit', ascending=False).iloc[0]
            
            # Advanced: Segment-based pricing
            segment_based_pricing = {}
            
            # Try optimizing price by different segments
            segment_factors = [
                'City', 'Type_of_vehicle', 'Weatherconditions', 
                'Road_traffic_density', 'IsWeekend', 'IsPeakHour'
            ]
            
            for factor in segment_factors:
                if factor in df.columns:
                    factor_pricing = {}
                    
                    for segment in df[factor].unique():
                        segment_df = df[df[factor] == segment]
                        
                        if len(segment_df) >= 100:  # Only analyze segments with enough data
                            segment_results = []
                            
                            for price in price_points:
                                profit = calculate_profit_at_price(price, segment_df)
                                avg_profit = profit.mean()
                                profit_margin = avg_profit / price * 100
                                profitable_pct = (profit > 0).mean() * 100
                                
                                segment_results.append({
                                    'Base_Price': price,
                                    'Average_Profit': avg_profit,
                                    'Profit_Margin_Pct': profit_margin,
                                    'Profitable_Deliveries_Pct': profitable_pct
                                })
                            
                            segment_analysis = pd.DataFrame(segment_results)
                            viable_prices = segment_analysis[segment_analysis['Profitable_Deliveries_Pct'] >= 90]
                            
                            if len(viable_prices) > 0:
                                segment_optimal = viable_prices.sort_values('Average_Profit', ascending=False).iloc[0]
                            else:
                                segment_optimal = segment_analysis.sort_values('Average_Profit', ascending=False).iloc[0]
                            
                            factor_pricing[segment] = {
                                'optimal_price': segment_optimal['Base_Price'],
                                'average_profit': segment_optimal['Average_Profit'],
                                'profit_margin': segment_optimal['Profit_Margin_Pct'],
                                'profitable_pct': segment_optimal['Profitable_Deliveries_Pct'],
                                'sample_size': len(segment_df)
                            }
                    
                    segment_based_pricing[factor] = factor_pricing
            
            # Dynamic pricing model
            # Train a model to predict the optimal price based on order characteristics
            
            # Create feature set for price prediction
            price_features = [
                'Distance_km', 'Time_taken(min)', 'Wait_Time_Minutes', 
                'Order_Hour', 'IsWeekend', 'IsPeakHour', 'IsHoliday',
                'Delivery_Speed', 'Restaurant_Prep_Time'
            ]
            available_features = [f for f in price_features if f in df.columns]
            
            if len(available_features) >= 5:  # Only proceed if we have enough features
                X = df[available_features].fillna(0)
                # Target is the total cost + desired profit margin (25%)
                y = df['Total_Cost'] * 1.25
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=RANDOM_STATE
                )
                
                # Train model
                dynamic_price_model = xgb.XGBRegressor(
                    n_estimators=100, 
                    learning_rate=0.1,
                    max_depth=5,
                    random_state=RANDOM_STATE
                )
                dynamic_price_model.fit(X_train, y_train)
                
                # Evaluate model
                y_pred = dynamic_price_model.predict(X_test)
                price_model_mae = mean_absolute_error(y_test, y_pred)
                price_model_mse = mean_squared_error(y_test, y_pred)
                price_model_r2 = r2_score(y_test, y_pred)
                
                # Feature importance
                price_feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': dynamic_price_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                # Save the model
                self.models['dynamic_pricing'] = dynamic_price_model
                self.encoders['dynamic_pricing_features'] = available_features
            else:
                dynamic_price_model = None
                price_model_mae = None
                price_model_mse = None
                price_model_r2 = None
                price_feature_importance = None
            
            # Store results
            self.results['pricing'] = {
                'price_analysis': price_analysis,
                'uniform_optimal_price': uniform_optimal_price,
                'segment_based_pricing': segment_based_pricing,
                'dynamic_pricing_model': {
                    'model': dynamic_price_model,
                    'mae': price_model_mae,
                    'mse': price_model_mse,
                    'r2': price_model_r2,
                    'feature_importance': price_feature_importance
                }
            }
            
            logger.info("Pricing optimization complete")
            return True
        
        except Exception as e:
            logger.error(f"Error in pricing optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def predict_delivery_time(self):
        """Train models to predict delivery time."""
        if not hasattr(self, 'df_cleaned'):
            logger.error("No preprocessed data available. Run preprocess_data first.")
            return False
        
        logger.info("Starting delivery time prediction modeling")
        try:
            df = self.df_cleaned.copy()
            
            # Features for time prediction
            time_features = [
                'Distance_km', 'Order_Hour', 'Pickup_Hour', 'DayOfWeek', 
                'IsWeekend', 'IsPeakHour', 'IsHoliday', 'Weatherconditions',
                'Road_traffic_density', 'Wait_Time_Minutes', 'City', 'Type_of_vehicle'
            ]
            
            # Filter available features
            available_features = [f for f in time_features if f in df.columns]
            
            if len(available_features) < 5:
                logger.error("Insufficient features for time prediction model")
                return False
                
            # Prepare data
            X = df[available_features].copy()
            y = df['Time_taken(min)']
            
            # Handle categorical variables
            cat_features = ['Weatherconditions', 'Road_traffic_density', 'City', 'Type_of_vehicle']
            cat_features = [f for f in cat_features if f in available_features]
            num_features = [f for f in available_features if f not in cat_features]
            
            # Create preprocessing pipeline
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), num_features),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
                ]
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=RANDOM_STATE
            )
            
            # Try multiple models
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=10, 
                    random_state=RANDOM_STATE
                ),
                'XGBoost': xgb.XGBRegressor(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=5, 
                    random_state=RANDOM_STATE
                ),
                'LightGBM': lgb.LGBMRegressor(
                    n_estimators=100, 
                    learning_rate=0.1, 
                    max_depth=5, 
                    random_state=RANDOM_STATE
                )
            }
            
            # Create pipelines
            model_pipelines = {
                name: Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ]) for name, model in models.items()
            }
            
            # Evaluate models
            model_results = {}
            best_score = float('inf')
            best_model_name = None
            
            for name, pipeline in model_pipelines.items():
                logger.info(f"Training {name}...")
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                model_results[name] = {
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R2': r2
                }
                
                if mae < best_score:
                    best_score = mae
                    best_model_name = name
                
                logger.info(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")
            
            # Feature importance for best model
            if best_model_name in ['Random Forest', 'XGBoost', 'LightGBM']:
                if best_model_name == 'Random Forest':
                    # Get feature names from preprocessor
                    cat_feature_names = preprocessor.transformers_[1][1].get_feature_names_out(cat_features)
                    feature_names = np.concatenate([num_features, cat_feature_names])
                    
                    # Get feature importances
                    best_model = model_pipelines[best_model_name]['model']
                    feature_importances = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                else:
                    # For XGBoost and LightGBM, get feature importance directly
                    best_model = model_pipelines[best_model_name]['model']
                    feature_importances = pd.DataFrame({
                        'Feature': best_model.feature_names_in_,
                        'Importance': best_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
            else:
                feature_importances = None
            
            # Save results
            self.models['time_prediction'] = {
                'best_model': model_pipelines[best_model_name],
                'best_model_name': best_model_name,
                'all_models': model_pipelines,
                'feature_names': available_features,
                'cat_features': cat_features,
                'num_features': num_features,
                'preprocessor': preprocessor
            }
            
            self.results['time_prediction'] = {
                'model_results': model_results,
                'best_model_name': best_model_name,
                'best_score': best_score,
                'feature_importance': feature_importances
            }
            
            logger.info(f"Time prediction modeling complete. Best model: {best_model_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error in time prediction modeling: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def demand_forecasting(self):
        """Forecast future delivery demand using time series analysis."""
        if not hasattr(self, 'df_cleaned'):
            logger.error("No preprocessed data available. Run preprocess_data first.")
            return False
        
        logger.info("Starting demand forecasting")
        try:
            df = self.df_cleaned.copy()
            
            # Create time series of order volume
            if 'Order_Date' in df.columns and 'Order_Datetime' in df.columns:
                # Daily forecasting
                daily_orders = df.groupby(df['Order_Date'].dt.date).size()
                daily_orders = daily_orders.sort_index()
                daily_orders.index = pd.DatetimeIndex(daily_orders.index)
                
                # Replace any missing days with 0 orders
                all_days = pd.date_range(start=daily_orders.index.min(), end=daily_orders.index.max())
                daily_orders = daily_orders.reindex(all_days, fill_value=0)
                
                # Split data into train/test (last 14 days as test)
                train_data = daily_orders[:-14]
                test_data = daily_orders[-14:]
                
                # Train simple ARIMA model instead of auto_arima
                # Use (2,1,2) as default parameters which often work well for daily data
                try:
                    arima_model = ARIMA(train_data, order=(2,1,2))
                    arima_fit = arima_model.fit()
                    
                    # Make predictions
                    sarima_forecast = arima_fit.forecast(steps=len(test_data))
                    
                    # Evaluate model
                    sarima_mae = mean_absolute_error(test_data, sarima_forecast)
                    sarima_rmse = np.sqrt(mean_squared_error(test_data, sarima_forecast))
                    
                    # Forecast next 30 days
                    future_forecast = arima_fit.forecast(steps=30)
                    future_dates = pd.date_range(
                        start=daily_orders.index[-1] + pd.Timedelta(days=1),
                        periods=30
                    )
                    future_forecast = pd.Series(future_forecast, index=future_dates)
                except:
                    # If ARIMA fails, use simple moving average
                    logger.warning("ARIMA model failed, using moving average instead")
                    window_size = 7  # 7-day moving average
                    ma = train_data.rolling(window=window_size).mean()
                    ma = ma.fillna(train_data.mean())
                    
                    # Use last value for forecast
                    sarima_forecast = pd.Series([ma.iloc[-1]] * len(test_data), index=test_data.index)
                    sarima_mae = mean_absolute_error(test_data, sarima_forecast)
                    sarima_rmse = np.sqrt(mean_squared_error(test_data, sarima_forecast))
                    
                    # Forecast next 30 days (constant)
                    future_forecast = pd.Series([ma.iloc[-1]] * 30, 
                                            index=pd.date_range(start=daily_orders.index[-1] + pd.Timedelta(days=1), 
                                                            periods=30))
                    arima_fit = None
                
                # Hourly forecasting
                if len(df) >= 1000:  # Only if enough data
                    hourly_orders = df.groupby([
                        df['Order_Date'].dt.date, 
                        df['Order_Hour']
                    ]).size().reset_index()
                    hourly_orders.columns = ['date', 'hour', 'orders']
                    
                    # Create datetime index
                    hourly_orders['datetime'] = pd.to_datetime(hourly_orders['date']) + pd.to_timedelta(hourly_orders['hour'], unit='h')
                    hourly_orders = hourly_orders.set_index('datetime')['orders']
                    
                    # Train XGBoost model for hourly prediction
                    # Create features: hour, day of week, day of month, month
                    hourly_features = pd.DataFrame({
                        'hour': hourly_orders.index.hour,
                        'dayofweek': hourly_orders.index.dayofweek,
                        'dayofmonth': hourly_orders.index.day,
                        'month': hourly_orders.index.month,
                        'orders': hourly_orders.values
                    })
                    
                    # Add lag features (previous day, same hour)
                    for lag in [1, 7]:  # 1-day and 1-week lags
                        hourly_features[f'lag_{lag}d'] = hourly_features['orders'].shift(24 * lag)
                    
                    # Add moving averages
                    hourly_features['ma_7d'] = hourly_features['orders'].rolling(24 * 7).mean()
                    
                    # Drop NAs
                    hourly_features = hourly_features.dropna()
                    
                    # Train/test split
                    X = hourly_features.drop('orders', axis=1)
                    y = hourly_features['orders']
                    
                    # Use last 7 days as test
                    test_size = 24 * 7
                    X_train, X_test = X[:-test_size], X[-test_size:]
                    y_train, y_test = y[:-test_size], y[-test_size:]
                    
                    # Train model
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=RANDOM_STATE
                    )
                    xgb_model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = xgb_model.predict(X_test)
                    
                    # Evaluate
                    hourly_mae = mean_absolute_error(y_test, y_pred)
                    hourly_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    
                    # Feature importance
                    hourly_importance = pd.DataFrame({
                        'Feature': X.columns,
                        'Importance': xgb_model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    # Save hourly model
                    self.models['hourly_demand'] = xgb_model
                else:
                    hourly_mae = None
                    hourly_rmse = None
                    hourly_importance = None
                    xgb_model = None
                
                # Store results
                self.models['demand_forecast'] = arima_fit if arima_fit is not None else "Moving Average"
                
                self.results['demand_forecast'] = {
                    'daily_orders': daily_orders,
                    'sarima_forecast': sarima_forecast,
                    'future_forecast': future_forecast,
                    'sarima_mae': sarima_mae,
                    'sarima_rmse': sarima_rmse
                }
                
                if xgb_model is not None:
                    self.results['demand_forecast'].update({
                        'hourly_mae': hourly_mae,
                        'hourly_rmse': hourly_rmse,
                        'hourly_importance': hourly_importance
                    })
                
                logger.info("Demand forecasting complete")
                return True
            else:
                logger.error("Required date columns not found for demand forecasting")
                return False
                
        except Exception as e:
            logger.error(f"Error in demand forecasting: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def route_optimization(self):
        """Optimize delivery routes for efficiency."""
        if not hasattr(self, 'df_cleaned'):
            logger.error("No preprocessed data available. Run preprocess_data first.")
            return False
        
        logger.info("Starting route optimization")
        try:
            df = self.df_cleaned.copy()
            
            # Check if we have the necessary location data
            location_cols = [
                'Restaurant_latitude', 'Restaurant_longitude',
                'Delivery_location_latitude', 'Delivery_location_longitude'
            ]
            
            if not all(col in df.columns for col in location_cols):
                logger.error("Location columns not found for route optimization")
                return False
            
            # Group by restaurant clusters
            if 'Restaurant_Cluster' in df.columns:
                cluster_metrics = df.groupby('Restaurant_Cluster').agg({
                    'Time_taken(min)': ['mean', 'median', 'std'],
                    'Distance_km': ['mean', 'median', 'std'],
                    'Profit': ['mean', 'sum'],
                    'Restaurant_latitude': 'mean',
                    'Restaurant_longitude': 'mean'
                })
                
                # Find the most efficient clusters (high profit/time ratio)
                cluster_metrics['Profit_per_minute'] = (
                    cluster_metrics[('Profit', 'mean')] / 
                    cluster_metrics[('Time_taken(min)', 'mean')]
                )
                
                # Sort by profitability
                efficient_clusters = cluster_metrics.sort_values('Profit_per_minute', ascending=False)
                
                # Create route efficiency scores
                if 'Route_Type' in df.columns:
                    route_metrics = df.groupby('Route_Type').agg({
                        'Time_taken(min)': ['mean', 'median', 'count'],
                        'Distance_km': 'mean',
                        'Profit': 'mean'
                    })
                    
                    # Calculate efficiency metrics
                    route_metrics['Speed_km_per_min'] = (
                        route_metrics[('Distance_km', 'mean')] / 
                        route_metrics[('Time_taken(min)', 'mean')]
                    )
                    
                    route_metrics['Profit_per_min'] = (
                        route_metrics[('Profit', 'mean')] / 
                        route_metrics[('Time_taken(min)', 'mean')]
                    )
                    
                    route_metrics['Profit_per_km'] = (
                        route_metrics[('Profit', 'mean')] / 
                        route_metrics[('Distance_km', 'mean')]
                    )
                    
                    # Sort by profitability per minute
                    efficient_routes = route_metrics.sort_values('Profit_per_min', ascending=False)
                else:
                    efficient_routes = None
            else:
                efficient_clusters = None
                efficient_routes = None
            
            # Hot zone identification
            # Get coordinates for orders
            pickup_coords = df.values
            dropoff_coords = df.values
            
            # Identify hot zones using DBSCAN
            pickup_clustering = DBSCAN(eps=0.01, min_samples=10).fit(pickup_coords)
            dropoff_clustering = DBSCAN(eps=0.01, min_samples=10).fit(dropoff_coords)
            
            pickup_labels = pickup_clustering.labels_
            dropoff_labels = dropoff_clustering.labels_
            
            # Calculate metrics for each hot zone
            pickup_clusters = pd.DataFrame({
                'cluster': pickup_labels,
                'latitude': pickup_coords[:, 0],
                'longitude': pickup_coords[:, 1]
            })
            
            dropoff_clusters = pd.DataFrame({
                'cluster': dropoff_labels,
                'latitude': dropoff_coords[:, 0],
                'longitude': dropoff_coords[:, 1]
            })
            
            # Get cluster centers and counts
            pickup_hotspots = pickup_clusters[pickup_clusters['cluster'] >= 0].groupby('cluster').agg({
                'latitude': 'mean',
                'longitude': 'mean',
                'cluster': 'count'
            }).rename(columns={'cluster': 'count'}).reset_index()
            
            dropoff_hotspots = dropoff_clusters[dropoff_clusters['cluster'] >= 0].groupby('cluster').agg({
                'latitude': 'mean',
                'longitude': 'mean',
                'cluster': 'count'
            }).rename(columns={'cluster': 'count'}).reset_index()
            
            # Save the results
            self.results['route_optimization'] = {
                'efficient_clusters': efficient_clusters,
                'efficient_routes': efficient_routes,
                'pickup_hotspots': pickup_hotspots,
                'dropoff_hotspots': dropoff_hotspots
            }
            
            logger.info("Route optimization complete")
            return True
        
        except Exception as e:
            logger.error(f"Error in route optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def driver_allocation(self):
        """Optimize driver allocation and scheduling."""
        if not hasattr(self, 'df_cleaned'):
            logger.error("No preprocessed data available. Run preprocess_data first.")
            return False
        
        logger.info("Starting driver allocation optimization")
        try:
            df = self.df_cleaned.copy()
            
            # Analyze delivery volume by hour and day
            if 'Order_Hour' in df.columns and 'DayOfWeek' in df.columns:
                # Create hourly demand patterns
                hourly_demand = df.groupby(['DayOfWeek', 'Order_Hour']).size().unstack(fill_value=0)
                
                # Calculate required drivers per hour
                # Assuming a driver can handle 2 deliveries per hour
                deliveries_per_driver_hour = 2
                drivers_needed = hourly_demand / deliveries_per_driver_hour
                drivers_needed = drivers_needed.round().astype(int)
                
                # Calculate driver utilization
                if 'Delivery_person_ID' in df.columns:
                    driver_metrics = df.groupby('Delivery_person_ID').agg({
                        'Order_ID': 'count',
                        'Time_taken(min)': ['sum', 'mean'],
                        'Distance_km': ['sum', 'mean'],
                        'Profit': ['sum', 'mean']
                    })
                    
                    # Calculate efficiency metrics
                    driver_metrics['Deliveries_per_day'] = driver_metrics[('Order_ID', 'count')] / df['Order_Date'].nunique()
                    driver_metrics['Profit_per_minute'] = driver_metrics[('Profit', 'sum')] / driver_metrics[('Time_taken(min)', 'sum')]
                    driver_metrics['Profit_per_km'] = driver_metrics[('Profit', 'sum')] / driver_metrics[('Distance_km', 'sum')]
                    
                    # Sort by profitability
                    efficient_drivers = driver_metrics.sort_values(('Profit_per_minute'), ascending=False)
                    
                    # Analyze driver ratings if available
                    if 'Delivery_person_Ratings' in df.columns:
                        driver_ratings = df.groupby('Delivery_person_ID')['Delivery_person_Ratings'].agg(['mean', 'count'])
                        
                        # Get high-rated drivers with sufficient deliveries
                        min_deliveries = 10
                        high_rated_drivers = driver_ratings[driver_ratings['count'] >= min_deliveries].sort_values('mean', ascending=False)
                    else:
                        high_rated_drivers = None
                else:
                    efficient_drivers = None
                    high_rated_drivers = None
                
                # Driver shift optimization
                # Create a simplified shift allocation model
                days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                shifts = [
                    'Morning (6 AM - 2 PM)',
                    'Afternoon (2 PM - 10 PM)',
                    'Night (10 PM - 6 AM)'
                ]
                
                # Map hours to shifts
                def map_hour_to_shift(hour):
                    if 6 <= hour < 14:
                        return 0  # Morning
                    elif 14 <= hour < 22:
                        return 1  # Afternoon
                    else:
                        return 2  # Night
                
                # Create shift demand
                shift_demand = {}
                for day in range(7):
                    shift_demand[days[day]] = {}
                    for shift_idx, shift_name in enumerate(shifts):
                        if shift_idx == 0:  # Morning
                            hours = range(6, 14)
                        elif shift_idx == 1:  # Afternoon
                            hours = range(14, 22)
                        else:  # Night
                            hours = list(range(22, 24)) + list(range(0, 6))
                        
                        # Calculate total deliveries in this shift
                        try:
                            total_deliveries = sum(hourly_demand.loc[day, hour] for hour in hours if hour in hourly_demand.columns)
                        except:
                            total_deliveries = 0
                        
                        # Calculate drivers needed (assume 8-hour shift, 2 deliveries per hour)
                        drivers_needed = max(1, int(total_deliveries / (8 * deliveries_per_driver_hour)))
                        shift_demand[days[day]][shift_name] = drivers_needed
                
                # Convert to DataFrame for easier visualization
                shift_allocation = pd.DataFrame(shift_demand).T
                
                # Store results
                self.results['driver_allocation'] = {
                    'hourly_demand': hourly_demand,
                    'drivers_needed': drivers_needed,
                    'shift_allocation': shift_allocation
                }
                
                if efficient_drivers is not None:
                    self.results['driver_allocation']['efficient_drivers'] = efficient_drivers
                
                if high_rated_drivers is not None:
                    self.results['driver_allocation']['high_rated_drivers'] = high_rated_drivers
                
                logger.info("Driver allocation optimization complete")
                return True
            else:
                logger.error("Required columns not found for driver allocation")
                return False
                
        except Exception as e:
            logger.error(f"Error in driver allocation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def simulate_dynamic_pricing(self):
        """Simulate dynamic pricing strategies and their impact on profit."""
        if not hasattr(self, 'df_cleaned'):
            logger.error("No preprocessed data available. Run preprocess_data first.")
            return False
        
        logger.info("Starting dynamic pricing simulation")
        try:
            df = self.df_cleaned.copy()
            
            # Define factors that influence dynamic pricing
            dynamic_factors = {
                'IsPeakHour': 1.5,       # 50% surge during peak hours
                'IsWeekend': 1.2,        # 20% surge on weekends
                'IsHoliday': 1.3,        # 30% surge on holidays
                'Road_traffic_density': {
                    'Jam': 1.4,           # 40% surge in high traffic
                    'High': 1.3,          # 30% surge in high traffic
                    'Medium': 1.15,       # 15% surge in medium traffic
                    'Low': 1.0            # No surge in low traffic
                },
                'Weatherconditions': {
                    'Fog': 1.3,           # 30% surge in fog
                    'Stormy': 1.4,        # 40% surge in storms
                    'Sandstorms': 1.35,   # 35% surge in sandstorms
                    'Windy': 1.2,         # 20% surge in windy conditions
                    'Cloudy': 1.1,        # 10% surge in cloudy conditions
                    'Sunny': 1.0          # No surge in sunny conditions
                },
                'Hour': {                 # Hour-specific multipliers
                    0: 1.2, 1: 1.3, 2: 1.4, 3: 1.5, 4: 1.4, 5: 1.3,
                    6: 1.1, 7: 1.2, 8: 1.3, 9: 1.1, 10: 1.0, 11: 1.1,
                    12: 1.2, 13: 1.1, 14: 1.0, 15: 1.0, 16: 1.1, 17: 1.2,
                    18: 1.3, 19: 1.4, 20: 1.3, 21: 1.2, 22: 1.1, 23: 1.2
                }
            }
            
            # Calculate base delivery fee (from preprocessing)
            base_fee = self.business_params['base_delivery_fee']
            
            # Apply dynamic pricing factors
            df['Dynamic_Multiplier'] = 1.0
            
            # Apply peak hour factor
            if 'IsPeakHour' in df.columns:
                df.loc[df['IsPeakHour'] == 1, 'Dynamic_Multiplier'] *= dynamic_factors['IsPeakHour']
            
            # Apply weekend factor
            if 'IsWeekend' in df.columns:
                df.loc[df['IsWeekend'] == 1, 'Dynamic_Multiplier'] *= dynamic_factors['IsWeekend']
            
            # Apply holiday factor
            if 'IsHoliday' in df.columns:
                df.loc[df['IsHoliday'] == 1, 'Dynamic_Multiplier'] *= dynamic_factors['IsHoliday']

            # Apply traffic factor
            if 'Road_traffic_density' in df.columns:
                for traffic_level, multiplier in dynamic_factors['Road_traffic_density'].items():
                    df.loc[df['Road_traffic_density'] == traffic_level, 'Dynamic_Multiplier'] *= multiplier
            
            # Apply weather factor
            if 'Weatherconditions' in df.columns:
                for weather_cond, multiplier in dynamic_factors['Weatherconditions'].items():
                    df.loc[df['Weatherconditions'] == weather_cond, 'Dynamic_Multiplier'] *= multiplier
            
            # Apply hour-specific factor
            if 'Order_Hour' in df.columns:
                for hour, multiplier in dynamic_factors['Hour'].items():
                    df.loc[df['Order_Hour'] == hour, 'Dynamic_Multiplier'] *= multiplier
            
            # Calculate dynamic price and new profit
            df['Dynamic_Price'] = base_fee * df['Dynamic_Multiplier']
            df['Dynamic_Profit'] = df['Dynamic_Price'] - df['Total_Cost']
            
            # Calculate profitability metrics
            static_profit = df['Profit'].sum()
            dynamic_profit = df['Dynamic_Profit'].sum()
            profit_increase = dynamic_profit - static_profit
            profit_increase_pct = (profit_increase / abs(static_profit)) * 100
            
            # Compare profitable orders
            static_profitable_pct = (df['Profit'] > 0).mean() * 100
            dynamic_profitable_pct = (df['Dynamic_Profit'] > 0).mean() * 100
            
            # Analyze price elasticity (simplified)
            # Group by price ranges and calculate order volume
            df['Price_Range'] = pd.cut(df['Dynamic_Price'], bins=10)
            price_elasticity = df.groupby('Price_Range').size()
            
            # Compare different pricing strategies
            pricing_strategies = {
                'static': {
                    'total_profit': static_profit,
                    'avg_profit_per_order': df['Profit'].mean(),
                    'profitable_orders_pct': static_profitable_pct
                },
                'dynamic': {
                    'total_profit': dynamic_profit,
                    'avg_profit_per_order': df['Dynamic_Profit'].mean(),
                    'profitable_orders_pct': dynamic_profitable_pct,
                    'profit_increase': profit_increase,
                    'profit_increase_pct': profit_increase_pct
                }
            }
            
            # Store results
            self.results['dynamic_pricing'] = {
                'pricing_strategies': pricing_strategies,
                'price_elasticity': price_elasticity,
                'dynamic_factors': dynamic_factors,
                'price_multipliers': df['Dynamic_Multiplier'].describe()
            }
            
            logger.info(f"Dynamic pricing simulation complete. Profit increase: {profit_increase_pct:.2f}%")
            return True
            
        except Exception as e:
            logger.error(f"Error in dynamic pricing simulation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def customer_segmentation(self):
        """Segment customers based on their ordering patterns."""
        if not hasattr(self, 'df_cleaned'):
            logger.error("No preprocessed data available. Run preprocess_data first.")
            return False
            
        logger.info("Starting customer segmentation analysis")
        try:
            df = self.df_cleaned.copy()
            
            # Check if we have customer ID
            if 'Customer_ID' not in df.columns:
                logger.error("Customer_ID column not found. Cannot perform customer segmentation.")
                return False
                
            # Aggregate customer data
            customer_data = df.groupby('Customer_ID').agg({
                'Order_ID': 'count',                       # Order frequency
                'Profit': ['sum', 'mean'],                 # Total and average profit
                'Distance_km': ['mean', 'max'],            # Average and max distance
                'Time_taken(min)': ['mean', 'sum'],        # Average and total time
                'Delivery_person_Ratings': 'mean',         # Average rating given
                'Restaurant_ID': lambda x: x.nunique(),    # Number of unique restaurants
                'Type_of_order': lambda x: x.mode().iloc[0] if not x.mode().empty else None  # Most common order type
            })
            
            # Rename columns for clarity
            customer_data.columns = [
                'order_count', 'total_profit', 'avg_profit_per_order', 
                'avg_distance', 'max_distance', 'avg_delivery_time', 
                'total_delivery_time', 'avg_rating', 'unique_restaurants',
                'most_common_order_type'
            ]
            
            # Calculate recency, frequency, monetary (RFM)
            if 'Order_Datetime' in df.columns:
                # Get the last order date for each customer
                last_order = df.groupby('Customer_ID')['Order_Datetime'].max()
                # Calculate days since last order
                max_date = df['Order_Datetime'].max()
                recency = (max_date - last_order).dt.days
                customer_data['days_since_last_order'] = recency
                
                # Calculate days between first and last order
                first_order = df.groupby('Customer_ID')['Order_Datetime'].min()
                customer_data['customer_lifetime_days'] = (last_order - first_order).dt.days
                
                # Calculate order frequency (orders per month)
                # Add small value to avoid division by zero
                customer_data['orders_per_month'] = customer_data['order_count'] / (customer_data['customer_lifetime_days'] / 30 + 0.01)
                
            # Perform customer segmentation using KMeans
            # Select features for clustering
            cluster_features = [
                'order_count', 'avg_profit_per_order', 'avg_distance',
                'avg_delivery_time', 'unique_restaurants'
            ]
            
            # Add RFM features if available
            if 'days_since_last_order' in customer_data.columns:
                cluster_features.extend(['days_since_last_order', 'orders_per_month'])
                
            # Drop customers with missing values for clustering
            cluster_data = customer_data[cluster_features].dropna()
            
            # Standardize features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Determine optimal number of clusters
            wcss = []
            max_clusters = min(10, len(scaled_data) - 1)  # Don't try more clusters than data points
            for i in range(1, max_clusters + 1):
                kmeans = KMeans(n_clusters=i, random_state=RANDOM_STATE, n_init=10)
                kmeans.fit(scaled_data)
                wcss.append(kmeans.inertia_)
                
            # Find elbow point (simplified method)
            # Calculate first derivative of WCSS
            if len(wcss) >= 3:  # Need at least 3 points for a sensible elbow
                derivatives = np.diff(wcss)
                # Look for the point where the derivative starts to flatten out
                rates_of_change = np.diff(derivatives)
                # Choose the point where the rate of change is minimized
                optimal_n_clusters = np.argmin(np.abs(rates_of_change)) + 2
            else:
                optimal_n_clusters = 3  # Default if not enough data
                
            # Apply KMeans with optimal clusters
            kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=RANDOM_STATE, n_init=10)
            cluster_data['Cluster'] = kmeans.fit_predict(scaled_data)
            
            # Analyze clusters
            cluster_analysis = cluster_data.groupby('Cluster').mean()
            
            # Get cluster sizes
            cluster_sizes = cluster_data['Cluster'].value_counts().to_dict()
            
            # Map clusters back to original customer data
            customer_cluster_map = dict(zip(cluster_data.index, cluster_data['Cluster']))
            customer_data['Cluster'] = customer_data.index.map(customer_cluster_map)
            
            # Name clusters based on characteristics
            # Higher value = higher index (0 is lowest)
            value_rank = customer_data.groupby('Cluster')['total_profit'].mean().rank(ascending=False)
            frequency_rank = customer_data.groupby('Cluster')['order_count'].mean().rank(ascending=False)
            
            # Create cluster names based on value and frequency
            cluster_names = {}
            for cluster in value_rank.index:
                v_rank = value_rank[cluster]
                f_rank = frequency_rank[cluster]
                
                if v_rank <= optimal_n_clusters / 2 and f_rank <= optimal_n_clusters / 2:
                    name = "High Value, High Frequency"
                elif v_rank <= optimal_n_clusters / 2:
                    name = "High Value, Low Frequency"
                elif f_rank <= optimal_n_clusters / 2:
                    name = "Low Value, High Frequency"
                else:
                    name = "Low Value, Low Frequency"
                    
                cluster_names[cluster] = f"Cluster {cluster} - {name}"
                
            # Store the results
            self.models['customer_segmentation'] = {
                'kmeans': kmeans,
                'scaler': scaler,
                'cluster_features': cluster_features
            }
            
            self.results['customer_segmentation'] = {
                'customer_data': customer_data,
                'cluster_analysis': cluster_analysis,
                'cluster_sizes': cluster_sizes,
                'cluster_names': cluster_names,
                'optimal_n_clusters': optimal_n_clusters,
                'wcss': wcss
            }
            
            logger.info(f"Customer segmentation complete with {optimal_n_clusters} clusters")
            return True
            
        except Exception as e:
            logger.error(f"Error in customer segmentation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def delivery_efficiency_analysis(self):
        """Analyze delivery efficiency and bottlenecks."""
        if not hasattr(self, 'df_cleaned'):
            logger.error("No preprocessed data available. Run preprocess_data first.")
            return False
            
        logger.info("Starting delivery efficiency analysis")
        try:
            df = self.df_cleaned.copy()
            
            # Calculate key performance indicators
            kpis = {
                'avg_delivery_time': df['Time_taken(min)'].mean(),
                'median_delivery_time': df['Time_taken(min)'].median(),
                'avg_distance': df['Distance_km'].mean(),
                'avg_speed_kmph': (df['Distance_km'] / (df['Time_taken(min)'] / 60)).mean(),
                'on_time_delivery_rate': 30  # Placeholder - would need actual target times to calculate
            }
            
            # Analyze delivery efficiency by various factors
            efficiency_factors = [
                'City', 'Type_of_vehicle', 'Weatherconditions', 
                'Road_traffic_density', 'IsWeekend', 'IsPeakHour'
            ]
            
            efficiency_by_factor = {}
            for factor in efficiency_factors:
                if factor in df.columns:
                    efficiency_by_factor[factor] = df.groupby(factor).agg({
                        'Time_taken(min)': ['mean', 'median', 'std'],
                        'Distance_km': 'mean',
                        'Delivery_Speed': ['mean', 'std']
                    })
                    
                    # Calculate average speed in km/h
                    efficiency_by_factor[factor]['avg_speed_kmph'] = (
                        efficiency_by_factor[factor][('Distance_km', 'mean')] / 
                        (efficiency_by_factor[factor][('Time_taken(min)', 'mean')] / 60)
                    )
            
            # Identify bottlenecks
            # Calculate average wait times
            bottlenecks = {
                'avg_wait_time': df['Wait_Time_Minutes'].mean(),
                'restaurant_prep_time': df['Restaurant_Prep_Time'].mean(),
                'restaurant_efficiency': df.groupby('Restaurant_ID')['Restaurant_Efficiency'].mean().sort_values()
            }
            
            # Time delay analysis
            if 'Order_Datetime' in df.columns and 'Pickup_Datetime' in df.columns:
                # Calculate time between ordering and pickup
                df['order_to_pickup_minutes'] = (df['Pickup_Datetime'] - df['Order_Datetime']).dt.total_seconds() / 60
                
                # Analyze time delays by hour of day
                time_delays = df.groupby('Order_Hour')['order_to_pickup_minutes'].agg(['mean', 'median', 'std'])
                
                # Analyze time delays by restaurant
                restaurant_delays = df.groupby('Restaurant_ID')['order_to_pickup_minutes'].agg(['mean', 'median', 'count'])
                restaurant_delays = restaurant_delays[restaurant_delays['count'] >= 10].sort_values('mean', ascending=False)
                
                bottlenecks['time_delays_by_hour'] = time_delays
                bottlenecks['restaurant_delays'] = restaurant_delays
            
            # Store results
            self.results['efficiency'] = {
                'kpis': kpis,
                'efficiency_by_factor': efficiency_by_factor,
                'bottlenecks': bottlenecks
            }
            
            logger.info("Delivery efficiency analysis complete")
            return True
            
        except Exception as e:
            logger.error(f"Error in delivery efficiency analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def generate_recommendations(self):
        """Generate business recommendations based on all analyses."""
        logger.info("Generating business recommendations")
        
        recommendations = {
            'pricing': [],
            'operations': [],
            'driver_management': [],
            'customer_strategies': [],
            'route_optimization': []
        }
        
        try:
            # Generate pricing recommendations
            if 'pricing' in self.results:
                pricing_data = self.results['pricing']
                
                # Check if we have optimal price data
                if 'uniform_optimal_price' in pricing_data:
                    optimal_price = pricing_data['uniform_optimal_price']['Base_Price']
                    recommendations['pricing'].append(
                        f"Set base delivery fee to {optimal_price} currency units for maximizing profit"
                    )
                
                # Check if we have segment-based pricing
                if 'segment_based_pricing' in pricing_data:
                    for factor, pricing in pricing_data['segment_based_pricing'].items():
                        top_segments = []
                        for segment, details in pricing.items():
                            if details['optimal_price'] > 0:
                                top_segments.append(f"{segment}: {details['optimal_price']}")
                        
                        if top_segments:
                            recommendations['pricing'].append(
                                f"Implement segment-based pricing for {factor}: " + ", ".join(top_segments[:3])
                            )
                
                # Add dynamic pricing recommendation
                if 'dynamic_pricing_model' in pricing_data and pricing_data['dynamic_pricing_model']['model'] is not None:
                    recommendations['pricing'].append(
                        "Implement dynamic pricing model based on distance, time, and demand patterns"
                    )
            
            # Generate operational recommendations
            if 'efficiency' in self.results:
                efficiency_data = self.results['efficiency']
                
                # Add recommendations based on bottlenecks
                if 'bottlenecks' in efficiency_data:
                    bottlenecks = efficiency_data['bottlenecks']
                    
                    if 'restaurant_delays' in bottlenecks and len(bottlenecks['restaurant_delays']) > 0:
                        # Get top 3 restaurants with longest delays
                        problem_restaurants = bottlenecks['restaurant_delays'].head(3).index.tolist()
                        recommendations['operations'].append(
                            f"Work with restaurants {', '.join(map(str, problem_restaurants))} to reduce preparation time"
                        )
                    
                    recommendations['operations'].append(
                        f"Focus on reducing average wait time ({bottlenecks['avg_wait_time']:.1f} minutes) by optimizing restaurant assignments"
                    )
                
                # Add recommendations based on efficiency factors
                if 'efficiency_by_factor' in efficiency_data:
                    factors = efficiency_data['efficiency_by_factor']
                    
                    # Find the factor with the most variance
                    factor_variances = {}
                    for factor_name, factor_data in factors.items():
                        if ('Time_taken(min)', 'std') in factor_data.columns:
                            factor_variances[factor_name] = factor_data[('Time_taken(min)', 'std')].mean()
                    
                    if factor_variances:
                        highest_variance_factor = max(factor_variances, key=factor_variances.get)
                        recommendations['operations'].append(
                            f"Address delivery time variability in {highest_variance_factor} to improve consistency"
                        )
            
            # Generate driver allocation recommendations
            if 'driver_allocation' in self.results:
                driver_data = self.results['driver_allocation']
                
                # Add recommendations based on shift allocation
                if 'shift_allocation' in driver_data:
                    shift_alloc = driver_data['shift_allocation']
                    
                    # Find peak days/shifts
                    peak_day = shift_alloc.sum(axis=1).idxmax()
                    peak_shifts = {}
                    for day in shift_alloc.index:
                        peak_shifts[day] = shift_alloc.loc[day].idxmax()
                    
                    recommendations['driver_management'].append(
                        f"Increase driver allocation for {peak_day}, especially during {peak_shifts[peak_day]}"
                    )
                
                # Add recommendations based on efficient drivers
                if 'efficient_drivers' in driver_data and driver_data['efficient_drivers'] is not None:
                    recommendations['driver_management'].append(
                        "Analyze top performing drivers' patterns to develop training program for new drivers"
                    )
                    
                    recommendations['driver_management'].append(
                        "Implement performance-based incentives for drivers to improve delivery efficiency"
                    )
            
            # Generate customer strategy recommendations
            if 'customer_segmentation' in self.results:
                customer_data = self.results['customer_segmentation']
                
                if 'cluster_names' in customer_data:
                    cluster_names = customer_data['cluster_names']
                    high_value_clusters = [c for c, name in cluster_names.items() if "High Value" in name]
                    
                    if high_value_clusters:
                        recommendations['customer_strategies'].append(
                            f"Develop loyalty program for high-value customer segments (clusters {', '.join(map(str, high_value_clusters))})"
                        )
                    
                    low_freq_clusters = [c for c, name in cluster_names.items() if "Low Frequency" in name]
                    if low_freq_clusters:
                        recommendations['customer_strategies'].append(
                            f"Target win-back campaigns for low-frequency customers (clusters {', '.join(map(str, low_freq_clusters))})"
                        )
            
            # Generate route optimization recommendations
            if 'route_optimization' in self.results:
                route_data = self.results['route_optimization']
                
                if 'efficient_routes' in route_data and route_data['efficient_routes'] is not None:
                    # Get top 3 efficient routes
                    top_routes = route_data['efficient_routes'].head(3).index.tolist()
                    
                    recommendations['route_optimization'].append(
                        f"Focus on high-profit route types: {', '.join(map(str, top_routes))}"
                    )
                
                if 'pickup_hotspots' in route_data and len(route_data['pickup_hotspots']) > 0:
                    recommendations['route_optimization'].append(
                        "Establish dark kitchens in high-density pickup areas to reduce delivery times and costs"
                    )
                
                if 'dropoff_hotspots' in route_data and len(route_data['dropoff_hotspots']) > 0:
                    recommendations['route_optimization'].append(
                        "Optimize driver starting positions based on delivery hotspots to minimize empty miles"
                    )
            
            # Store recommendations
            self.results['recommendations'] = recommendations
            
            logger.info("Business recommendations generated")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'error': str(e),
                'pricing': ["Unable to generate pricing recommendations due to an error."],
                'operations': ["Unable to generate operational recommendations due to an error."],
                'driver_management': ["Unable to generate driver management recommendations due to an error."],
                'customer_strategies': ["Unable to generate customer strategy recommendations due to an error."],
                'route_optimization': ["Unable to generate route optimization recommendations due to an error."]
            }
    
    def create_visualization(self, plot_type, **kwargs):
        """Create visualizations for analysis results."""
        try:
            plt.figure(figsize=(12, 8))
            
            if plot_type == 'profit_by_factor':
                factor = kwargs.get('factor', 'City')
                if 'profitability' in self.results:
                    profit_data = self.results['profitability']['profit_by_factor'].get(factor)
                    if profit_data is not None:
                        profit_means = profit_data[('Profit', 'mean')].sort_values(ascending=False)
                        
                        # Create bar chart
                        ax = sns.barplot(x=profit_means.index, y=profit_means.values)
                        plt.title(f'Average Profit by {factor}')
                        plt.xlabel(factor)
                        plt.ylabel('Average Profit')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        
                        # Add profit values on top of bars
                        for i, v in enumerate(profit_means.values):
                            ax.text(i, v + 1, f'{v:.2f}', ha='center')
                        
                        # Save plot
                        plot_path = f'profit_by_{factor}.png'
                        plt.savefig(plot_path)
                        plt.close()
                        
                        return plot_path
                    
            elif plot_type == 'profit_by_hour':
                if 'profitability' in self.results:
                    profit_hour = self.results['profitability']['profit_by_hour']
                    if profit_hour is not None:
                        profit_means = profit_hour[('Profit', 'mean')]
                        
                        # Create line chart
                        plt.plot(profit_means.index, profit_means.values, marker='o', linestyle='-', linewidth=2)
                        plt.title('Average Profit by Hour of Day')
                        plt.xlabel('Hour of Day')
                        plt.ylabel('Average Profit')
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        
                        # Save plot
                        plot_path = 'profit_by_hour.png'
                        plt.savefig(plot_path)
                        plt.close()
                        
                        return plot_path
                        
            elif plot_type == 'demand_forecast':
                if 'demand_forecast' in self.results:
                    forecast_data = self.results['demand_forecast']
                    if 'daily_orders' in forecast_data and 'future_forecast' in forecast_data:
                        daily_orders = forecast_data['daily_orders']
                        future_forecast = forecast_data['future_forecast']
                        
                        # Create time series plot
                        plt.plot(daily_orders.index, daily_orders.values, label='Historical Demand', color='blue')
                        plt.plot(future_forecast.index, future_forecast.values, label='Forecast', color='red', linestyle='--')
                        plt.title('Demand Forecast')
                        plt.xlabel('Date')
                        plt.ylabel('Number of Orders')
                        plt.legend()
                        plt.grid(True, linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        
                        # Save plot
                        plot_path = 'demand_forecast.png'
                        plt.savefig(plot_path)
                        plt.close()
                        
                        return plot_path
                        
            elif plot_type == 'cluster_map':
                if ('route_optimization' in self.results and 
                    'pickup_hotspots' in self.results['route_optimization'] and 
                    'dropoff_hotspots' in self.results['route_optimization']):
                    
                    pickup_spots = self.results['route_optimization']['pickup_hotspots']
                    dropoff_spots = self.results['route_optimization']['dropoff_hotspots']
                    
                    if len(pickup_spots) > 0 and len(dropoff_spots) > 0:
                        # Calculate center of the map
                        center_lat = (pickup_spots['latitude'].mean() + dropoff_spots['latitude'].mean()) / 2
                        center_lon = (pickup_spots['longitude'].mean() + dropoff_spots['longitude'].mean()) / 2
                        
                        # Create folium map
                        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
                        
                        # Add pickup clusters
                        pickup_cluster = MarkerCluster(name="Restaurant Clusters")
                        for _, row in pickup_spots.iterrows():
                            folium.Marker(
                                location=[row['latitude'], row['longitude']],
                                popup=f"Restaurant Cluster {row['cluster']}: {row['count']} orders",
                                icon=folium.Icon(color='green', icon='utensils', prefix='fa')
                            ).add_to(pickup_cluster)
                        pickup_cluster.add_to(m)
                        
                        # Add delivery clusters
                        delivery_cluster = MarkerCluster(name="Delivery Clusters")
                        for _, row in dropoff_spots.iterrows():
                            folium.Marker(
                                location=[row['latitude'], row['longitude']],
                                popup=f"Delivery Cluster {row['cluster']}: {row['count']} orders",
                                icon=folium.Icon(color='red', icon='home', prefix='fa')
                            ).add_to(delivery_cluster)
                        delivery_cluster.add_to(m)
                        
                        # Add layer control
                        folium.LayerControl().add_to(m)
                        
                        # Save map
                        map_path = 'cluster_map.html'
                        m.save(map_path)
                        
                        return map_path

            # Return None if plot couldn't be created
            return None
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def save_results(self, output_dir='./results'):
        """Save analysis results to files."""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Save preprocessed data
            if hasattr(self, 'df_preprocessed'):
                self.df_preprocessed.to_csv(f'{output_dir}/preprocessed_data.csv', index=False)
            
            if hasattr(self, 'df_cleaned'):
                self.df_cleaned.to_csv(f'{output_dir}/cleaned_data.csv', index=False)
            
            # Save models
            if self.models:
                os.makedirs(f'{output_dir}/models', exist_ok=True)
                for model_name, model in self.models.items():
                    try:
                        joblib.dump(model, f'{output_dir}/models/{model_name}.pkl')
                    except:
                        logger.warning(f"Could not save model {model_name}")
            
            # Save results as JSON (where possible)
            if self.results:
                os.makedirs(f'{output_dir}/results', exist_ok=True)
                for result_name, result in self.results.items():
                    try:
                        # Convert pandas DataFrames to CSV
                        if isinstance(result, pd.DataFrame):
                            result.to_csv(f'{output_dir}/results/{result_name}.csv')
                        elif isinstance(result, dict):
                            # Handle nested dictionaries with DataFrames
                            for key, value in result.items():
                                if isinstance(value, pd.DataFrame):
                                    value.to_csv(f'{output_dir}/results/{result_name}_{key}.csv')
                    except:
                        logger.warning(f"Could not save result {result_name}")
            
            logger.info(f"Results saved to {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return False
    
    def run_all_analyses(self):
        """Run all analyses in sequence."""
        try:
            logger.info("Starting comprehensive food delivery analysis")
            
            # Step 1: Preprocess data
            preprocess_success = self.preprocess_data()
            if not preprocess_success:
                logger.error("Preprocessing failed. Aborting analysis.")
                return False
                
            # Step 2: Run all analyses
            analysis_steps = [
                ('Profitability Analysis', self.analyze_profitability),
                ('Pricing Optimization', self.pricing_optimization),
                ('Delivery Time Prediction', self.predict_delivery_time),
                ('Demand Forecasting', self.demand_forecasting),
                ('Route Optimization', self.route_optimization),
                ('Driver Allocation', self.driver_allocation),
                ('Dynamic Pricing Simulation', self.simulate_dynamic_pricing),
                ('Customer Segmentation', self.customer_segmentation),
                ('Delivery Efficiency Analysis', self.delivery_efficiency_analysis)
            ]
            
            results = {}
            for name, func in analysis_steps:
                logger.info(f"Running {name}")
                success = func()
                results[name] = success
                if not success:
                    logger.warning(f"{name} did not complete successfully")
            
            # Step 3: Generate recommendations
            recommendations = self.generate_recommendations()
            
            # Step 4: Create visualizations
            visualizations = {}
            viz_types = [
                'profit_by_factor', 'profit_by_hour', 'demand_forecast', 'cluster_map'
            ]
            
            for viz in viz_types:
                viz_path = self.create_visualization(viz, factor='City')
                if viz_path:
                    visualizations[viz] = viz_path
            
            # Return analysis summary
            analysis_summary = {
                'analytics_results': results,
                'recommendations': recommendations,
                'visualizations': visualizations
            }
            
            logger.info("Comprehensive analysis completed")
            return analysis_summary
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}