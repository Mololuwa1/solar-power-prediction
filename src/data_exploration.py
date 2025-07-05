import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")

def explore_dataset():
    """Comprehensive exploration of the solar panel dataset"""
    
    # Get all CSV files
    data_dir = '/home/ubuntu/upload/'
    all_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    print(f"Total number of files: {len(all_files)}")
    print("\n" + "="*50)
    
    # Categorize files
    power_files = [f for f in all_files if 'Inverter' not in f and any(weather not in f for weather in ['Wind', 'Visibility', 'Temperature', 'SeaLevelPressure', 'RelativeHumidity', 'Rainfall', 'Irradiance'])]
    inverter_files = [f for f in all_files if 'Inverter' in f]
    weather_files = [f for f in all_files if any(weather in f for weather in ['Wind', 'Visibility', 'Temperature', 'SeaLevelPressure', 'RelativeHumidity', 'Rainfall', 'Irradiance'])]
    
    print(f"Power generation files: {len(power_files)}")
    print(f"Inverter files: {len(inverter_files)}")
    print(f"Weather files: {len(weather_files)}")
    
    # Sample a few files from each category for detailed analysis
    sample_power_file = power_files[0] if power_files else None
    sample_inverter_file = inverter_files[0] if inverter_files else None
    sample_weather_file = weather_files[0] if weather_files else None
    
    print("\n" + "="*50)
    print("DETAILED FILE ANALYSIS")
    print("="*50)
    
    # Analyze power generation data
    if sample_power_file:
        print(f"\nAnalyzing power generation file: {os.path.basename(sample_power_file)}")
        df_power = pd.read_csv(sample_power_file)
        print(f"Shape: {df_power.shape}")
        print(f"Columns: {list(df_power.columns)}")
        print(f"Date range: {df_power['Time'].min()} to {df_power['Time'].max()}")
        print(f"Sample data:")
        print(df_power.head())
        
        # Check for missing values
        print(f"\nMissing values:")
        print(df_power.isnull().sum())
    
    # Analyze inverter data
    if sample_inverter_file:
        print(f"\n\nAnalyzing inverter file: {os.path.basename(sample_inverter_file)}")
        df_inverter = pd.read_csv(sample_inverter_file)
        print(f"Shape: {df_inverter.shape}")
        print(f"Columns: {list(df_inverter.columns)}")
        print(f"Date range: {df_inverter['Time'].min()} to {df_inverter['Time'].max()}")
        print(f"Sample data:")
        print(df_inverter.head())
        
        # Check for missing values
        print(f"\nMissing values:")
        print(df_inverter.isnull().sum())
    
    # Analyze weather data
    if sample_weather_file:
        print(f"\n\nAnalyzing weather file: {os.path.basename(sample_weather_file)}")
        df_weather = pd.read_csv(sample_weather_file)
        print(f"Shape: {df_weather.shape}")
        print(f"Columns: {list(df_weather.columns)}")
        print(f"Date range: {df_weather['Time'].min()} to {df_weather['Time'].max()}")
        print(f"Sample data:")
        print(df_weather.head())
        
        # Check for missing values
        print(f"\nMissing values:")
        print(df_weather.isnull().sum())
    
    print("\n" + "="*50)
    print("FILE INVENTORY")
    print("="*50)
    
    # Create inventory of all files
    file_inventory = []
    
    for file_path in all_files:
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path, nrows=1)  # Read just first row to get columns
            file_info = {
                'filename': filename,
                'type': 'power' if 'Inverter' not in filename and not any(w in filename for w in ['Wind', 'Visibility', 'Temperature', 'SeaLevelPressure', 'RelativeHumidity', 'Rainfall', 'Irradiance']) else 'inverter' if 'Inverter' in filename else 'weather',
                'columns': list(df.columns),
                'num_columns': len(df.columns)
            }
            file_inventory.append(file_info)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    
    # Group by type and show summary
    for file_type in ['power', 'inverter', 'weather']:
        type_files = [f for f in file_inventory if f['type'] == file_type]
        print(f"\n{file_type.upper()} FILES ({len(type_files)}):")
        for f in type_files[:10]:  # Show first 10
            print(f"  {f['filename']} - {f['num_columns']} columns")
        if len(type_files) > 10:
            print(f"  ... and {len(type_files) - 10} more files")
    
    return file_inventory

def analyze_weather_data():
    """Analyze weather data in detail"""
    print("\n" + "="*50)
    print("WEATHER DATA ANALYSIS")
    print("="*50)
    
    data_dir = '/home/ubuntu/upload/'
    
    # Weather variables
    weather_vars = ['Temperature', 'Irradiance', 'RelativeHumidity', 'Wind', 'Rainfall', 'SeaLevelPressure', 'Visibility']
    
    weather_summary = {}
    
    for var in weather_vars:
        var_files = glob.glob(os.path.join(data_dir, f'{var}_*.csv'))
        if var_files:
            print(f"\n{var}:")
            print(f"  Files: {[os.path.basename(f) for f in var_files]}")
            
            # Read one file to understand structure
            try:
                df = pd.read_csv(var_files[0])
                print(f"  Columns: {list(df.columns)}")
                print(f"  Sample values: {df.iloc[0, 1] if len(df.columns) > 1 else 'N/A'}")
                weather_summary[var] = {
                    'files': len(var_files),
                    'columns': list(df.columns),
                    'sample_value': df.iloc[0, 1] if len(df.columns) > 1 else None
                }
            except Exception as e:
                print(f"  Error reading: {e}")
    
    return weather_summary

def analyze_power_stations():
    """Analyze power generation stations"""
    print("\n" + "="*50)
    print("POWER STATION ANALYSIS")
    print("="*50)
    
    data_dir = '/home/ubuntu/upload/'
    
    # Get power generation files (excluding weather and inverter)
    power_files = glob.glob(os.path.join(data_dir, '*.csv'))
    power_files = [f for f in power_files if 'Inverter' not in f and not any(weather in f for weather in ['Wind', 'Visibility', 'Temperature', 'SeaLevelPressure', 'RelativeHumidity', 'Rainfall', 'Irradiance'])]
    
    print(f"Total power generation files: {len(power_files)}")
    
    station_summary = {}
    
    for file_path in power_files[:10]:  # Analyze first 10 files
        filename = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            
            # Convert time column
            df['Time'] = pd.to_datetime(df['Time'])
            
            station_info = {
                'filename': filename,
                'shape': df.shape,
                'date_range': (df['Time'].min(), df['Time'].max()),
                'columns': list(df.columns),
                'max_power': df['power(W)'].max() if 'power(W)' in df.columns else None,
                'total_generation': df['generation(kWh)'].sum() if 'generation(kWh)' in df.columns else None
            }
            
            station_summary[filename] = station_info
            
            print(f"\n{filename}:")
            print(f"  Shape: {station_info['shape']}")
            print(f"  Date range: {station_info['date_range'][0]} to {station_info['date_range'][1]}")
            print(f"  Max power: {station_info['max_power']} W")
            print(f"  Total generation: {station_info['total_generation']:.2f} kWh")
            
        except Exception as e:
            print(f"Error analyzing {filename}: {e}")
    
    return station_summary

if __name__ == "__main__":
    print("SOLAR PANEL DATASET EXPLORATION")
    print("="*50)
    
    # Run comprehensive analysis
    file_inventory = explore_dataset()
    weather_summary = analyze_weather_data()
    station_summary = analyze_power_stations()
    
    print("\n" + "="*50)
    print("EXPLORATION COMPLETE")
    print("="*50)
    print(f"Total files analyzed: {len(file_inventory)}")
    print("Next steps: Data preprocessing and feature engineering")

