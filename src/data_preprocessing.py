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
        station_name = filename.replace('.csv', '')\n        \n        try:\n        try:
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
            print(f"Error reading {filename}: {e}")\n    \n    if power_data_list:\n        power_data = pd.concat(power_data_list, ignore_index=True)\n        power_data = power_data.sort_values(['Station', 'Time'])\n        \n        print(f"Combined power data shape: {power_data.shape}")\n        print(f"Number of stations: {power_data['Station'].nunique()}")\n        print(f"Date range: {power_data['Time'].min()} to {power_data['Time'].max()}")\n        \n        return power_data\n    else:\n        print("No power generation data found!")\n        return None\n\ndef create_time_features(df, time_col='Time'):\n    """Create time-based features from datetime column"""\n    \n    print("Creating time-based features...")\n    \n    df = df.copy()\n    df[time_col] = pd.to_datetime(df[time_col])\n    \n    # Basic time features\n    df['Year'] = df[time_col].dt.year\n    df['Month'] = df[time_col].dt.month\n    df['Day'] = df[time_col].dt.day\n    df['Hour'] = df[time_col].dt.hour\n    df['DayOfWeek'] = df[time_col].dt.dayofweek\n    df['DayOfYear'] = df[time_col].dt.dayofyear\n    df['WeekOfYear'] = df[time_col].dt.isocalendar().week\n    \n    # Cyclical features (important for solar prediction)\n    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)\n    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)\n    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)\n    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)\n    df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)\n    df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)\n    \n    # Season feature\n    df['Season'] = df['Month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',\n                                   3: 'Spring', 4: 'Spring', 5: 'Spring',\n                                   6: 'Summer', 7: 'Summer', 8: 'Summer',\n                                   9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})\n    \n    # Is weekend\n    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)\n    \n    # Solar elevation angle approximation (simplified)\n    # This is a rough approximation - in practice, you'd use more precise calculations\n    df['SolarElevation'] = np.maximum(0, \n        np.sin(np.radians(23.45 * np.sin(2 * np.pi * (df['DayOfYear'] - 81) / 365))) * \n        np.sin(np.radians(22.3)) +  # Approximate latitude for Hong Kong\n        np.cos(np.radians(23.45 * np.sin(2 * np.pi * (df['DayOfYear'] - 81) / 365))) * \n        np.cos(np.radians(22.3)) * \n        np.cos(2 * np.pi * (df['Hour'] - 12) / 24))\n    \n    print(f"Added {len([col for col in df.columns if col not in [time_col]]) - len([col for col in df.columns if col not in [time_col, 'Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', 'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos', 'Season', 'IsWeekend', 'SolarElevation']])} time-based features")\n    \n    return df\n\ndef create_lag_features(df, target_col, station_col, time_col='Time', lags=[1, 2, 3, 6, 12, 24]):\n    """Create lag features for time series prediction"""\n    \n    print(f"Creating lag features for {target_col}...")\n    \n    df = df.copy()\n    df = df.sort_values([station_col, time_col])\n    \n    for lag in lags:\n        lag_col = f'{target_col}_lag_{lag}'\n        df[lag_col] = df.groupby(station_col)[target_col].shift(lag)\n    \n    # Rolling statistics\n    for window in [6, 12, 24]:\n        df[f'{target_col}_rolling_mean_{window}'] = df.groupby(station_col)[target_col].transform(\n            lambda x: x.rolling(window=window, min_periods=1).mean())\n        df[f'{target_col}_rolling_std_{window}'] = df.groupby(station_col)[target_col].transform(\n            lambda x: x.rolling(window=window, min_periods=1).std())\n    \n    print(f"Added {len(lags)} lag features and {len([6, 12, 24]) * 2} rolling statistics")\n    \n    return df\n\ndef handle_missing_values(df):\n    """Handle missing values in the dataset"""\n    \n    print("Handling missing values...")\n    \n    # Check missing values\n    missing_before = df.isnull().sum()\n    print(f"Missing values before handling:")\n    print(missing_before[missing_before > 0])\n    \n    # Forward fill for weather data (reasonable for short gaps)\n    weather_cols = [col for col in df.columns if any(weather in col for weather in \n                   ['Temperature', 'Irradiance', 'RelativeHumidity', 'Wind', 'Rainfall', 'SeaLevelPressure', 'Visibility'])]\n    \n    for col in weather_cols:\n        if col in df.columns:\n            df[col] = df[col].fillna(method='ffill').fillna(method='bfill')\n    \n    # For lag features, drop rows with missing values (they're at the beginning)\n    lag_cols = [col for col in df.columns if 'lag_' in col or 'rolling_' in col]\n    \n    # Fill remaining missing values with median for numeric columns\n    numeric_cols = df.select_dtypes(include=[np.number]).columns\n    for col in numeric_cols:\n        if df[col].isnull().sum() > 0:\n            df[col] = df[col].fillna(df[col].median())\n    \n    # Check missing values after handling\n    missing_after = df.isnull().sum()\n    print(f"Missing values after handling:")\n    print(missing_after[missing_after > 0])\n    \n    return df\n\ndef remove_outliers(df, target_col, method='iqr', threshold=3):\n    """Remove outliers from the dataset"""\n    \n    print(f"Removing outliers from {target_col}...")\n    \n    initial_size = len(df)\n    \n    if method == 'iqr':\n        Q1 = df[target_col].quantile(0.25)\n        Q3 = df[target_col].quantile(0.75)\n        IQR = Q3 - Q1\n        lower_bound = Q1 - 1.5 * IQR\n        upper_bound = Q3 + 1.5 * IQR\n        \n        df = df[(df[target_col] >= lower_bound) & (df[target_col] <= upper_bound)]\n        \n    elif method == 'zscore':\n        z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())\n        df = df[z_scores < threshold]\n    \n    final_size = len(df)\n    removed = initial_size - final_size\n    \n    print(f"Removed {removed} outliers ({removed/initial_size*100:.2f}% of data)")\n    \n    return df\n\ndef prepare_modeling_dataset():\n    """Main function to prepare the complete dataset for modeling"""\n    \n    print("="*50)\n    print("PREPARING MODELING DATASET")\n    print("="*50)\n    \n    # Load weather data\n    weather_data = load_and_merge_weather_data()\n    \n    # Load power generation data\n    power_data = load_power_generation_data()\n    \n    if weather_data is None or power_data is None:\n        print("Error: Could not load required data")\n        return None\n    \n    # Resample weather data to hourly (to match power data frequency)\n    print("Resampling weather data to hourly frequency...")\n    weather_data = weather_data.set_index('Time').resample('H').mean().reset_index()\n    \n    # Merge power and weather data\n    print("Merging power and weather data...")\n    \n    # Round time to nearest hour for better matching\n    power_data['Time_rounded'] = power_data['Time'].dt.round('H')\n    weather_data['Time_rounded'] = weather_data['Time'].dt.round('H')\n    \n    merged_data = pd.merge(power_data, weather_data, \n                          left_on='Time_rounded', right_on='Time_rounded', \n                          how='inner', suffixes=('', '_weather'))\n    \n    # Keep original time column\n    merged_data = merged_data.drop(['Time_weather'], axis=1)\n    merged_data = merged_data.rename(columns={'Time_rounded': 'Time'})\n    \n    print(f"Merged data shape: {merged_data.shape}")\n    \n    # Create time features\n    merged_data = create_time_features(merged_data)\n    \n    # Create lag features for power prediction\n    merged_data = create_lag_features(merged_data, 'Power_W', 'Station')\n    \n    # Handle missing values\n    merged_data = handle_missing_values(merged_data)\n    \n    # Remove outliers\n    merged_data = remove_outliers(merged_data, 'Power_W')\n    \n    # Create additional features\n    print("Creating additional features...")\n    \n    # Power density (power per unit irradiance)\n    merged_data['Power_Density'] = merged_data['Power_W'] / (merged_data['Irradiance'] + 1e-6)\n    \n    # Temperature efficiency factor (solar panels are less efficient at high temperatures)\n    merged_data['Temp_Efficiency'] = 1 - 0.004 * np.maximum(0, merged_data['Temperature'] - 25)\n    \n    # Clear sky index (actual irradiance vs theoretical maximum)\n    max_irradiance_by_hour = merged_data.groupby('Hour')['Irradiance'].transform('max')\n    merged_data['Clear_Sky_Index'] = merged_data['Irradiance'] / (max_irradiance_by_hour + 1e-6)\n    \n    # Station encoding (for model that needs numeric features)\n    from sklearn.preprocessing import LabelEncoder\n    le = LabelEncoder()\n    merged_data['Station_encoded'] = le.fit_transform(merged_data['Station'])\n    \n    print(f"Final dataset shape: {merged_data.shape}")\n    print(f"Features: {merged_data.columns.tolist()}")\n    \n    # Save the processed dataset\n    merged_data.to_csv('/home/ubuntu/processed_solar_data.csv', index=False)\n    print("Processed dataset saved to: /home/ubuntu/processed_solar_data.csv")\n    \n    # Save feature information\n    feature_info = {\n        'target_variable': 'Power_W',\n        'features': [col for col in merged_data.columns if col not in ['Power_W', 'Generation_kWh', 'Time', 'Station']],\n        'categorical_features': ['Season'],\n        'numeric_features': [col for col in merged_data.columns if col not in ['Power_W', 'Generation_kWh', 'Time', 'Station', 'Season']],\n        'time_features': ['Year', 'Month', 'Day', 'Hour', 'DayOfWeek', 'DayOfYear', 'WeekOfYear', \n                         'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos', 'IsWeekend', 'SolarElevation'],\n        'weather_features': [col for col in merged_data.columns if any(weather in col for weather in \n                           ['Temperature', 'Irradiance', 'RelativeHumidity', 'Wind', 'Rainfall', 'SeaLevelPressure', 'Visibility'])],\n        'lag_features': [col for col in merged_data.columns if 'lag_' in col or 'rolling_' in col],\n        'engineered_features': ['Power_Density', 'Temp_Efficiency', 'Clear_Sky_Index', 'Station_encoded']\n    }\n    \n    import json\n    with open('/home/ubuntu/feature_info.json', 'w') as f:\n        json.dump(feature_info, f, indent=2)\n    \n    print("Feature information saved to: /home/ubuntu/feature_info.json")\n    \n    return merged_data, feature_info\n\nif __name__ == "__main__":\n    # Prepare the modeling dataset\n    dataset, features = prepare_modeling_dataset()\n    \n    if dataset is not None:\n        print("\\n" + "="*50)\n        print("DATASET SUMMARY")\n        print("="*50)\n        print(f"Total samples: {len(dataset):,}")\n        print(f"Total features: {len(features['features'])}")\n        print(f"Date range: {dataset['Time'].min()} to {dataset['Time'].max()}")\n        print(f"Stations: {dataset['Station'].nunique()}")\n        print(f"Target variable statistics:")\n        print(dataset['Power_W'].describe())\n        \n        print("\\nDataset preparation complete!")

