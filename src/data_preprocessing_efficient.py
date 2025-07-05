import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_sample_data():
    """Load a representative sample of the data for modeling"""
    
    print("Loading sample data for efficient processing...")
    data_dir = '/home/ubuntu/upload/'
    
    # Load weather data (sample)
    print("Loading weather data...")
    weather_files = {
        'Temperature': '/home/ubuntu/upload/Temperature_2022.csv',
        'Irradiance': '/home/ubuntu/upload/Irradiance_2022.csv',
        'RelativeHumidity': '/home/ubuntu/upload/RelativeHumidity_2022.csv',
        'Wind': '/home/ubuntu/upload/Wind_2022.csv',
        'Rainfall': '/home/ubuntu/upload/Rainfall_2022.csv',
        'SeaLevelPressure': '/home/ubuntu/upload/SeaLevelPressure_2022.csv',
        'Visibility': '/home/ubuntu/upload/Visibility_2022.csv'
    }
    
    weather_data = None
    for var_name, file_path in weather_files.items():
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Time'] = pd.to_datetime(df['Time'])
            
            # Rename columns for consistency
            if var_name == 'Temperature':
                df = df.rename(columns={'Temp (Degree Celsius)': 'Temperature'})
            elif var_name == 'Irradiance':
                df = df.rename(columns={'Irradiance (W/m2)': 'Irradiance'})
            elif var_name == 'RelativeHumidity':
                df = df.rename(columns={'RH (%)': 'RelativeHumidity'})
            elif var_name == 'Wind':
                df = df.rename(columns={'Wind Speed (m/s)': 'WindSpeed', 'Wind Direction (degree)': 'WindDirection'})
            elif var_name == 'Rainfall':
                df = df.rename(columns={'Rainfall(mm)': 'Rainfall'})
            elif var_name == 'SeaLevelPressure':
                df = df.rename(columns={'SLP (hPa)': 'SeaLevelPressure'})
            elif var_name == 'Visibility':
                df = df.rename(columns={'Vis (km)': 'Visibility'})
            
            if weather_data is None:
                weather_data = df
            else:
                weather_data = pd.merge(weather_data, df, on='Time', how='outer')
    
    # Resample to hourly
    weather_data = weather_data.set_index('Time').resample('H').mean().reset_index()
    
    # Load power data (sample from a few representative stations)
    print("Loading power generation data...")
    sample_stations = [
        '/home/ubuntu/upload/ZoneL1(Station1).csv',
        '/home/ubuntu/upload/ZoneL1(Station2).csv',
        '/home/ubuntu/upload/SQ14.csv',
        '/home/ubuntu/upload/SQ18.csv',
        '/home/ubuntu/upload/UGHall4.csv'
    ]
    
    power_data_list = []
    for file_path in sample_stations:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Time'] = pd.to_datetime(df['Time'])
            df['Station'] = os.path.basename(file_path).replace('.csv', '')
            
            # Standardize column names
            if 'power(W)' in df.columns:
                df = df.rename(columns={'power(W)': 'Power_W'})
            if 'generation(kWh)' in df.columns:
                df = df.rename(columns={'generation(kWh)': 'Generation_kWh'})
            
            # Filter to 2022 data to match weather data
            df = df[(df['Time'] >= '2022-01-01') & (df['Time'] <= '2022-12-31')]
            
            power_data_list.append(df[['Time', 'Station', 'Power_W', 'Generation_kWh']])
    
    power_data = pd.concat(power_data_list, ignore_index=True)
    
    # Round time to nearest hour for merging
    power_data['Time'] = power_data['Time'].dt.round('H')
    weather_data['Time'] = weather_data['Time'].dt.round('H')
    
    # Merge data
    merged_data = pd.merge(power_data, weather_data, on='Time', how='inner')
    
    print(f"Sample dataset shape: {merged_data.shape}")
    print(f"Date range: {merged_data['Time'].min()} to {merged_data['Time'].max()}")
    print(f"Stations: {merged_data['Station'].nunique()}")
    
    return merged_data

def create_features(df):
    """Create all features for the model"""
    
    print("Creating features...")
    
    # Time features
    df['Hour'] = df['Time'].dt.hour
    df['Month'] = df['Time'].dt.month
    df['DayOfWeek'] = df['Time'].dt.dayofweek
    df['DayOfYear'] = df['Time'].dt.dayofyear
    
    # Cyclical features
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    
    # Season
    df['Season'] = df['Month'].map({12: 0, 1: 0, 2: 0,  # Winter
                                   3: 1, 4: 1, 5: 1,    # Spring
                                   6: 2, 7: 2, 8: 2,    # Summer
                                   9: 3, 10: 3, 11: 3}) # Autumn
    
    # Weekend indicator
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Solar elevation (simplified)
    df['SolarElevation'] = np.maximum(0, 
        np.sin(np.radians(23.45 * np.sin(2 * np.pi * (df['DayOfYear'] - 81) / 365))) * 
        np.sin(np.radians(22.3)) +  # Hong Kong latitude
        np.cos(np.radians(23.45 * np.sin(2 * np.pi * (df['DayOfYear'] - 81) / 365))) * 
        np.cos(np.radians(22.3)) * 
        np.cos(2 * np.pi * (df['Hour'] - 12) / 24))
    
    # Weather-based features
    df['Power_Density'] = df['Power_W'] / (df['Irradiance'] + 1e-6)
    df['Temp_Efficiency'] = 1 - 0.004 * np.maximum(0, df['Temperature'] - 25)
    
    # Clear sky index
    max_irradiance_by_hour = df.groupby('Hour')['Irradiance'].transform('max')
    df['Clear_Sky_Index'] = df['Irradiance'] / (max_irradiance_by_hour + 1e-6)
    
    # Station encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['Station_encoded'] = le.fit_transform(df['Station'])
    
    # Simple lag features (only a few to avoid memory issues)
    df = df.sort_values(['Station', 'Time'])
    df['Power_lag_1'] = df.groupby('Station')['Power_W'].shift(1)
    df['Power_lag_24'] = df.groupby('Station')['Power_W'].shift(24)
    
    # Rolling mean (6 hour window)
    df['Power_rolling_mean_6'] = df.groupby('Station')['Power_W'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean())
    
    print(f"Created features. Final shape: {df.shape}")
    
    return df

def prepare_model_data():
    """Prepare data for modeling"""
    
    print("="*50)
    print("PREPARING SAMPLE DATASET FOR MODELING")
    print("="*50)
    
    # Load sample data
    data = load_sample_data()
    
    # Create features
    data = create_features(data)
    
    # Handle missing values
    print("Handling missing values...")
    
    # Fill missing values
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())
    
    # Remove outliers (simple method)
    print("Removing outliers...")
    Q1 = data['Power_W'].quantile(0.25)
    Q3 = data['Power_W'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    initial_size = len(data)
    data = data[(data['Power_W'] >= lower_bound) & (data['Power_W'] <= upper_bound)]
    final_size = len(data)
    
    print(f"Removed {initial_size - final_size} outliers ({(initial_size - final_size)/initial_size*100:.2f}%)")
    
    # Save processed data
    data.to_csv('/home/ubuntu/processed_solar_sample.csv', index=False)
    
    # Define features for modeling
    feature_cols = [col for col in data.columns if col not in ['Time', 'Station', 'Power_W', 'Generation_kWh']]
    
    feature_info = {
        'target': 'Power_W',
        'features': feature_cols,
        'categorical': ['Season'],
        'numeric': [col for col in feature_cols if col != 'Season']
    }
    
    import json
    with open('/home/ubuntu/feature_info_sample.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"Final dataset shape: {data.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Target variable statistics:")
    print(data['Power_W'].describe())
    
    return data, feature_info

if __name__ == "__main__":
    dataset, features = prepare_model_data()
    print("\nSample dataset preparation complete!")
    print("Files saved:")
    print("- /home/ubuntu/processed_solar_sample.csv")
    print("- /home/ubuntu/feature_info_sample.json")

