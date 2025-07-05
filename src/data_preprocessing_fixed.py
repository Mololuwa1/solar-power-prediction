import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_merge_weather_data():
    """Load and merge all weather data into a single DataFrame"""
    
    print("Loading weather data...")
    data_dir = '/home/ubuntu/upload/'
    
    # Weather variables and their files
    weather_vars = {
        'Temperature': 'Temp (Degree Celsius)',
        'Irradiance': 'Irradiance (W/m2)',
        'RelativeHumidity': 'RH (%)',
        'Wind': ['Wind Speed (m/s)', 'Wind Direction (degree)'],
        'Rainfall': 'Rainfall(mm)',
        'SeaLevelPressure': 'SLP (hPa)',
        'Visibility': 'Vis (km)'
    }
    
    weather_dfs = []
    
    for var_name, col_names in weather_vars.items():
        var_files = glob.glob(os.path.join(data_dir, f'{var_name}_*.csv'))
        
        if var_files:
            print(f"Processing {var_name} files: {len(var_files)}")
            
            var_data = []
            for file in var_files:
                try:
                    df = pd.read_csv(file)
                    df['Time'] = pd.to_datetime(df['Time'])
                    var_data.append(df)
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            
            if var_data:
                # Concatenate all years for this variable
                combined_var = pd.concat(var_data, ignore_index=True)
                combined_var = combined_var.sort_values('Time').drop_duplicates(subset=['Time'])
                
                # Rename columns for consistency
                if isinstance(col_names, list):
                    # For wind data with multiple columns
                    rename_dict = {'Time': 'Time'}
                    for i, col in enumerate(col_names):
                        if col in combined_var.columns:
                            rename_dict[col] = f'{var_name}_{col.split("(")[0].strip().replace(" ", "_")}'
                else:
                    # For single column variables
                    if col_names in combined_var.columns:
                        combined_var = combined_var.rename(columns={col_names: var_name})
                
                weather_dfs.append(combined_var)
    
    # Merge all weather data
    if weather_dfs:
        weather_data = weather_dfs[0]
        for df in weather_dfs[1:]:
            weather_data = pd.merge(weather_data, df, on='Time', how='outer')
        
        weather_data = weather_data.sort_values('Time')
        print(f"Combined weather data shape: {weather_data.shape}")
        print(f"Date range: {weather_data['Time'].min()} to {weather_data['Time'].max()}")
        
        return weather_data
    else:
        print("No weather data found!")
        return None

def load_power_generation_data():
    """Load power generation data from multiple stations"""
    
    print("Loading power generation data...")
    data_dir = '/home/ubuntu/upload/'
    
    # Get power generation files (excluding weather and inverter)
    power_files = glob.glob(os.path.join(data_dir, '*.csv'))
    power_files = [f for f in power_files if 'Inverter' not in f and not any(weather in f for weather in ['Wind', 'Visibility', 'Temperature', 'SeaLevelPressure', 'RelativeHumidity', 'Rainfall', 'Irradiance'])]
    
    print(f"Found {len(power_files)} power generation files")
    
    power_data_list = []
    
    for file_path in power_files:
        filename = os.path.basename(file_path)
        station_name = filename.replace('.csv', '')
        
        try:
            df = pd.read_csv(file_path)
            df['Time'] = pd.to_datetime(df['Time'])
            df['Station'] = station_name
            
            # Standardize column names
            if 'power(W)' in df.columns:
                df = df.rename(columns={'power(W)': 'Power_W'})
            if 'generation(kWh)' in df.columns:
                df = df.rename(columns={'generation(kWh)': 'Generation_kWh'})
            
            power_data_list.append(df[['Time', 'Station', 'Power_W', 'Generation_kWh']])
            
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    if power_data_list:
        power_data = pd.concat(power_data_list, ignore_index=True)
        power_data = power_data.sort_values(['Station', 'Time'])
        
        print(f"Combined power data shape: {power_data.shape}")
        print(f"Number of stations: {power_data['Station'].nunique()}")
        print(f"Date range: {power_data['Time'].min()} to {power_data['Time'].max()}")
        
        return power_data
    else:
        print("No power generation data found!")
        return None

def create_time_features(df, time_col='Time'):
    """Create time-based features from datetime column"""
    
    print("Creating time-based features...")
    
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Basic time features
    df['Year'] = df[time_col].dt.year
    df['Month'] = df[time_col].dt.month
    df['Day'] = df[time_col].dt.day
    df['Hour'] = df[time_col].dt.hour
    df['DayOfWeek'] = df[time_col].dt.dayofweek
    df['DayOfYear'] = df[time_col].dt.dayofyear
    df['WeekOfYear'] = df[time_col].dt.isocalendar().week
    
    # Cyclical features (important for solar prediction)
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    
    # Season feature
    df['Season'] = df['Month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                   3: 'Spring', 4: 'Spring', 5: 'Spring',
                                   6: 'Summer', 7: 'Summer', 8: 'Summer',
                                   9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
    
    # Is weekend
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
    
    # Solar elevation angle approximation (simplified)
    df['SolarElevation'] = np.maximum(0, 
        np.sin(np.radians(23.45 * np.sin(2 * np.pi * (df['DayOfYear'] - 81) / 365))) * 
        np.sin(np.radians(22.3)) +  # Approximate latitude for Hong Kong
        np.cos(np.radians(23.45 * np.sin(2 * np.pi * (df['DayOfYear'] - 81) / 365))) * 
        np.cos(np.radians(22.3)) * 
        np.cos(2 * np.pi * (df['Hour'] - 12) / 24))
    
    print("Added time-based features")
    
    return df

def create_lag_features(df, target_col, station_col, time_col='Time', lags=[1, 2, 3, 6, 12, 24]):
    """Create lag features for time series prediction"""
    
    print(f"Creating lag features for {target_col}...")
    
    df = df.copy()
    df = df.sort_values([station_col, time_col])
    
    for lag in lags:
        lag_col = f'{target_col}_lag_{lag}'
        df[lag_col] = df.groupby(station_col)[target_col].shift(lag)
    
    # Rolling statistics
    for window in [6, 12, 24]:
        df[f'{target_col}_rolling_mean_{window}'] = df.groupby(station_col)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f'{target_col}_rolling_std_{window}'] = df.groupby(station_col)[target_col].transform(
            lambda x: x.rolling(window=window, min_periods=1).std())
    
    print(f"Added {len(lags)} lag features and {len([6, 12, 24]) * 2} rolling statistics")
    
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    
    print("Handling missing values...")
    
    # Check missing values
    missing_before = df.isnull().sum()
    print(f"Missing values before handling:")
    print(missing_before[missing_before > 0])
    
    # Forward fill for weather data (reasonable for short gaps)
    weather_cols = [col for col in df.columns if any(weather in col for weather in 
                   ['Temperature', 'Irradiance', 'RelativeHumidity', 'Wind', 'Rainfall', 'SeaLevelPressure', 'Visibility'])]
    
    for col in weather_cols:
        if col in df.columns:
            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Fill remaining missing values with median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # Check missing values after handling
    missing_after = df.isnull().sum()
    print(f"Missing values after handling:")
    print(missing_after[missing_after > 0])
    
    return df

def remove_outliers(df, target_col, method='iqr', threshold=3):
    """Remove outliers from the dataset"""
    
    print(f"Removing outliers from {target_col}...")
    
    initial_size = len(df)
    
    if method == 'iqr':
        Q1 = df[target_col].quantile(0.25)
        Q3 = df[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]
        
    elif method == 'zscore':
        z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
        df = df[z_scores < threshold]
    
    final_size = len(df)
    removed = initial_size - final_size
    
    print(f"Removed {removed} outliers ({removed/initial_size*100:.2f}% of data)")
    
    return df

def prepare_modeling_dataset():
    """Main function to prepare the complete dataset for modeling"""
    
    print("="*50)
    print("PREPARING MODELING DATASET")
    print("="*50)
    
    # Load weather data
    weather_data = load_and_merge_weather_data()
    
    # Load power generation data
    power_data = load_power_generation_data()
    
    if weather_data is None or power_data is None:
        print("Error: Could not load required data")
        return None, None
    
    # Resample weather data to hourly (to match power data frequency)
    print("Resampling weather data to hourly frequency...")
    weather_data = weather_data.set_index('Time').resample('H').mean().reset_index()
    
    # Merge power and weather data
    print("Merging power and weather data...")
    
    # Round time to nearest hour for better matching
    power_data['Time_rounded'] = power_data['Time'].dt.round('H')
    weather_data['Time_rounded'] = weather_data['Time'].dt.round('H')
    
    # Drop original Time columns before merge to avoid duplicates
    power_data_merge = power_data.drop('Time', axis=1)
    weather_data_merge = weather_data.drop('Time', axis=1)
    
    merged_data = pd.merge(power_data_merge, weather_data_merge, 
                          on='Time_rounded', 
                          how='inner')
    
    # Rename Time_rounded to Time
    merged_data = merged_data.rename(columns={'Time_rounded': 'Time'})
    
    print(f"Merged data shape: {merged_data.shape}")
    print(f"Columns: {merged_data.columns.tolist()}")
    
    # Create time features
    merged_data = create_time_features(merged_data)
    
    # Create lag features for power prediction
    merged_data = create_lag_features(merged_data, 'Power_W', 'Station')
    
    # Handle missing values
    merged_data = handle_missing_values(merged_data)
    
    # Remove outliers
    merged_data = remove_outliers(merged_data, 'Power_W')
    
    # Create additional features
    print("Creating additional features...")
    
    # Power density (power per unit irradiance)
    merged_data['Power_Density'] = merged_data['Power_W'] / (merged_data['Irradiance'] + 1e-6)
    
    # Temperature efficiency factor (solar panels are less efficient at high temperatures)
    merged_data['Temp_Efficiency'] = 1 - 0.004 * np.maximum(0, merged_data['Temperature'] - 25)
    
    # Clear sky index (actual irradiance vs theoretical maximum)
    max_irradiance_by_hour = merged_data.groupby('Hour')['Irradiance'].transform('max')
    merged_data['Clear_Sky_Index'] = merged_data['Irradiance'] / (max_irradiance_by_hour + 1e-6)
    
    # Station encoding (for model that needs numeric features)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    merged_data['Station_encoded'] = le.fit_transform(merged_data['Station'])
    
    print(f"Final dataset shape: {merged_data.shape}")
    
    # Save the processed dataset
    merged_data.to_csv('/home/ubuntu/processed_solar_data.csv', index=False)
    print("Processed dataset saved to: /home/ubuntu/processed_solar_data.csv")
    
    # Save feature information
    feature_info = {
        'target_variable': 'Power_W',
        'features': [col for col in merged_data.columns if col not in ['Power_W', 'Generation_kWh', 'Time', 'Station']],
        'categorical_features': ['Season'],
        'numeric_features': [col for col in merged_data.columns if col not in ['Power_W', 'Generation_kWh', 'Time', 'Station', 'Season']],
        'time_features': ['Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', 
                         'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos', 'IsWeekend', 'SolarElevation'],
        'weather_features': [col for col in merged_data.columns if any(weather in col for weather in 
                           ['Temperature', 'Irradiance', 'RelativeHumidity', 'Wind', 'Rainfall', 'SeaLevelPressure', 'Visibility'])],
        'lag_features': [col for col in merged_data.columns if 'lag_' in col or 'rolling_' in col],
        'engineered_features': ['Power_Density', 'Temp_Efficiency', 'Clear_Sky_Index', 'Station_encoded']
    }
    
    import json
    with open('/home/ubuntu/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print("Feature information saved to: /home/ubuntu/feature_info.json")
    
    return merged_data, feature_info

if __name__ == "__main__":
    # Prepare the modeling dataset
    dataset, features = prepare_modeling_dataset()
    
    if dataset is not None:
        print("\n" + "="*50)
        print("DATASET SUMMARY")
        print("="*50)
        print(f"Total samples: {len(dataset):,}")
        print(f"Total features: {len(features['features'])}")
        print(f"Date range: {dataset['Time'].min()} to {dataset['Time'].max()}")
        print(f"Stations: {dataset['Station'].nunique()}")
        print(f"Target variable statistics:")
        print(dataset['Power_W'].describe())
        
        print("\nDataset preparation complete!")

