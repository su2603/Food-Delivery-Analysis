# Save this as run_food_delivery.py

import pandas as pd
import numpy as np
import logging
import os
from foodDeliveryAnalysis import FoodDeliveryAnalysis

# Set up logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("food_delivery_analysis.log"),
        logging.StreamHandler()
    ]
)

# Main function to run the analysis
def main():
    print("Starting Food Delivery Analysis...")
    
    # Create sample data if no data file exists
    data_file = 'sample_food_delivery_data.csv'
    
    if not os.path.exists(data_file):
        print(f"Creating sample data file: {data_file}")
        # Create a simple sample dataset
        np.random.seed(42)
        num_samples = 1000
        
        data = {
            'Order_ID': [f'ORD-{i:05d}' for i in range(1, num_samples+1)],
            'Customer_ID': [f'CUST-{np.random.randint(1, 200):03d}' for _ in range(num_samples)],
            'Restaurant_ID': [f'REST-{np.random.randint(1, 50):02d}' for _ in range(num_samples)],
            'Delivery_person_ID': [f'DP-{np.random.randint(1, 30):02d}' for _ in range(num_samples)],
            'Order_Date': pd.date_range(start='2023-01-01', periods=100).repeat(10)[:num_samples].strftime('%d-%m-%Y'),
            'Time_Orderd': [f'{np.random.randint(10, 22):02d}:{np.random.choice([0, 15, 30, 45]):02d}' for _ in range(num_samples)],
            'Time_Order_picked': [f'{np.random.randint(10, 22):02d}:{np.random.choice([0, 15, 30, 45]):02d}' for _ in range(num_samples)],
            'Weatherconditions': np.random.choice(['Sunny', 'Cloudy', 'Fog', 'Windy', 'Stormy'], num_samples),
            'Road_traffic_density': np.random.choice(['Low', 'Medium', 'High', 'Jam'], num_samples),
            'Type_of_vehicle': np.random.choice(['motorcycle', 'scooter', 'bicycle'], num_samples),
            'Type_of_order': np.random.choice(['Snack', 'Meal', 'Drinks', 'Buffet'], num_samples),
            'City': np.random.choice(['Urban', 'Metropolitian', 'Semi-Urban'], num_samples),
            'Restaurant_latitude': np.random.uniform(28.4, 28.7, num_samples),
            'Restaurant_longitude': np.random.uniform(77.0, 77.3, num_samples),
            'Delivery_location_latitude': np.random.uniform(28.4, 28.7, num_samples),
            'Delivery_location_longitude': np.random.uniform(77.0, 77.3, num_samples),
            'Delivery_person_Age': np.random.randint(18, 45, num_samples),
            'Delivery_person_Ratings': np.random.uniform(3, 5, num_samples).round(1),
            'Time_taken(min)': [f'{np.random.randint(15, 45)} min' for _ in range(num_samples)]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(data_file, index=False)
        print(f"Sample data created with {num_samples} records")
    
    # Initialize our analysis class
    print("Initializing FoodDeliveryAnalysis...")
    fda = FoodDeliveryAnalysis(data_path=data_file)
    
    # We need to fix the preprocessing function before running it
    # Let's patch the object
    def fixed_preprocess_data(self):
        """Fixed preprocess_data method to handle imputation correctly"""
        if self.df is None:
            logging.error("No data available. Load data first.")
            return False
        
        logging.info("Starting data preprocessing")
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
            logging.info("Calculating distances and geospatial features")
            df['Distance_km'] = df.apply(self._calculate_haversine_distance, axis=1)
            
            # Create straight-line vs actual path ratio (approximate)
            # We'll use a multiplier to simulate real-world routes which are rarely straight lines
            df['Route_Efficiency'] = np.random.uniform(1.0, 1.5, size=len(df))
            df['Actual_Distance_km'] = df['Distance_km'] * df['Route_Efficiency']
            
            # --- DateTime Processing ---
            logging.info("Processing datetime features")
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
            
            # Fix: Only impute the Wait_Time_Minutes column, not the entire dataframe
            from sklearn.impute import KNNImputer
            wait_time_imputer = KNNImputer(n_neighbors=5)
            wait_time_array = df['Wait_Time_Minutes'].values.reshape(-1, 1)  # Reshape for 2D array
            df['Wait_Time_Minutes'] = wait_time_imputer.fit_transform(wait_time_array)
            
            # --- Advanced Feature Engineering ---
            logging.info("Creating advanced features")
            
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
            logging.info("Calculating costs and profits")
            
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
            
            # Skip location clustering for simplicity in this demo
            
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
            
            logging.info(f"Preprocessing complete. Clean data shape: {df_cleaned.shape}")
            return True
        
        except Exception as e:
            logging.error(f"Error in preprocessing: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return False
    
    # Apply our fixed preprocessing method
    fda.preprocess_data = fixed_preprocess_data.__get__(fda, FoodDeliveryAnalysis)
    
    # Run preprocessing
    print("Running preprocessing...")
    if fda.preprocess_data():
        print("Preprocessing completed successfully!")
        
        # Run a few analyses
        print("Running profitability analysis...")
        fda.analyze_profitability()
        
        print("Running pricing optimization...")
        fda.pricing_optimization()
        
        print("Running demand forecasting...")
        fda.demand_forecasting()
        
        # Generate recommendations
        print("Generating recommendations...")
        recommendations = fda.generate_recommendations()
        
        # Print some results
        print("\n--- RECOMMENDATIONS ---")
        for category, rec_list in recommendations.items():
            print(f"\n{category.upper()}:")
            for rec in rec_list:
                print(f"- {rec}")
        
        # Save results
        print("\nSaving results...")
        fda.save_results()
        
        print("\nAnalysis complete!")
    else:
        print("Preprocessing failed.")

if __name__ == "__main__":
    main()