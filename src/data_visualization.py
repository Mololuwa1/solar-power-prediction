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
plt.rcParams['figure.figsize'] = (12, 8)

def create_initial_visualizations():
    """Create initial visualizations to understand data patterns"""
    
    data_dir = '/home/ubuntu/upload/'
    
    # Create output directory for plots
    os.makedirs('/home/ubuntu/plots', exist_ok=True)
    
    print("Creating initial data visualizations...")
    
    # 1. Power generation patterns
    print("1. Analyzing power generation patterns...")
    
    # Load a representative power generation file
    power_file = '/home/ubuntu/upload/ZoneL1(Station1).csv'
    df_power = pd.read_csv(power_file)
    df_power['Time'] = pd.to_datetime(df_power['Time'])
    df_power['Hour'] = df_power['Time'].dt.hour
    df_power['Month'] = df_power['Time'].dt.month
    df_power['DayOfYear'] = df_power['Time'].dt.dayofyear
    
    # Create subplots for power analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Power Generation Analysis - ZoneL1(Station1)', fontsize=16)
    
    # Daily power pattern
    hourly_avg = df_power.groupby('Hour')['power(W)'].mean()
    axes[0,0].plot(hourly_avg.index, hourly_avg.values, marker='o')
    axes[0,0].set_title('Average Power by Hour of Day')
    axes[0,0].set_xlabel('Hour')
    axes[0,0].set_ylabel('Power (W)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Monthly power pattern
    monthly_avg = df_power.groupby('Month')['power(W)'].mean()
    axes[0,1].bar(monthly_avg.index, monthly_avg.values)
    axes[0,1].set_title('Average Power by Month')
    axes[0,1].set_xlabel('Month')
    axes[0,1].set_ylabel('Power (W)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Power distribution
    axes[1,0].hist(df_power['power(W)'], bins=50, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Power Distribution')
    axes[1,0].set_xlabel('Power (W)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(True, alpha=0.3)
    
    # Time series plot (sample period)
    sample_data = df_power[df_power['Time'].dt.date == df_power['Time'].dt.date.iloc[1000]]
    axes[1,1].plot(sample_data['Time'], sample_data['power(W)'])
    axes[1,1].set_title('Sample Daily Power Pattern')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Power (W)')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/plots/power_generation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Weather data analysis
    print("2. Analyzing weather data patterns...")
    
    # Load weather data
    temp_file = '/home/ubuntu/upload/Temperature_2021.csv'
    irr_file = '/home/ubuntu/upload/Irradiance_2021.csv'
    
    df_temp = pd.read_csv(temp_file)
    df_irr = pd.read_csv(irr_file)
    
    df_temp['Time'] = pd.to_datetime(df_temp['Time'])
    df_irr['Time'] = pd.to_datetime(df_irr['Time'])
    
    # Sample data for visualization (every 60th point to reduce density)
    df_temp_sample = df_temp.iloc[::60].copy()
    df_irr_sample = df_irr.iloc[::60].copy()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Weather Data Analysis - 2021', fontsize=16)
    
    # Temperature time series
    axes[0,0].plot(df_temp_sample['Time'], df_temp_sample['Temp (Degree Celsius)'], alpha=0.7)
    axes[0,0].set_title('Temperature Over Time')
    axes[0,0].set_xlabel('Time')
    axes[0,0].set_ylabel('Temperature (°C)')
    axes[0,0].grid(True, alpha=0.3)
    
    # Irradiance time series
    axes[0,1].plot(df_irr_sample['Time'], df_irr_sample['Irradiance (W/m2)'], alpha=0.7, color='orange')
    axes[0,1].set_title('Solar Irradiance Over Time')
    axes[0,1].set_xlabel('Time')
    axes[0,1].set_ylabel('Irradiance (W/m²)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Temperature distribution
    axes[1,0].hist(df_temp['Temp (Degree Celsius)'], bins=50, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Temperature Distribution')
    axes[1,0].set_xlabel('Temperature (°C)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].grid(True, alpha=0.3)
    
    # Irradiance distribution
    axes[1,1].hist(df_irr['Irradiance (W/m2)'], bins=50, alpha=0.7, edgecolor='black', color='orange')
    axes[1,1].set_title('Irradiance Distribution')
    axes[1,1].set_xlabel('Irradiance (W/m²)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/plots/weather_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Station comparison
    print("3. Comparing different power stations...")
    
    # Load multiple station files
    station_files = [
        '/home/ubuntu/upload/ZoneL1(Station1).csv',
        '/home/ubuntu/upload/ZoneL1(Station2).csv',
        '/home/ubuntu/upload/SQ14.csv',
        '/home/ubuntu/upload/SQ18.csv'
    ]
    
    station_data = {}
    for file in station_files:
        if os.path.exists(file):
            station_name = os.path.basename(file).replace('.csv', '')
            df = pd.read_csv(file)
            df['Time'] = pd.to_datetime(df['Time'])
            station_data[station_name] = df
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Power Station Comparison', fontsize=16)
    
    # Max power comparison
    max_powers = [df['power(W)'].max() for df in station_data.values()]
    station_names = list(station_data.keys())
    
    axes[0,0].bar(range(len(station_names)), max_powers)
    axes[0,0].set_title('Maximum Power by Station')
    axes[0,0].set_xlabel('Station')
    axes[0,0].set_ylabel('Max Power (W)')
    axes[0,0].set_xticks(range(len(station_names)))
    axes[0,0].set_xticklabels(station_names, rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # Total generation comparison
    total_gen = [df['generation(kWh)'].sum() for df in station_data.values()]
    
    axes[0,1].bar(range(len(station_names)), total_gen)
    axes[0,1].set_title('Total Generation by Station')
    axes[0,1].set_xlabel('Station')
    axes[0,1].set_ylabel('Total Generation (kWh)')
    axes[0,1].set_xticks(range(len(station_names)))
    axes[0,1].set_xticklabels(station_names, rotation=45)
    axes[0,1].grid(True, alpha=0.3)
    
    # Power time series comparison (sample period)
    for i, (name, df) in enumerate(station_data.items()):
        sample_period = df[(df['Time'] >= '2022-06-01') & (df['Time'] <= '2022-06-07')]
        if not sample_period.empty:
            axes[1,0].plot(sample_period['Time'], sample_period['power(W)'], 
                          label=name, alpha=0.7)
    
    axes[1,0].set_title('Power Comparison (Sample Week)')
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Power (W)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Capacity factor comparison (average power / max power)
    capacity_factors = [df['power(W)'].mean() / df['power(W)'].max() * 100 
                       for df in station_data.values()]
    
    axes[1,1].bar(range(len(station_names)), capacity_factors)
    axes[1,1].set_title('Capacity Factor by Station')
    axes[1,1].set_xlabel('Station')
    axes[1,1].set_ylabel('Capacity Factor (%)')
    axes[1,1].set_xticks(range(len(station_names)))
    axes[1,1].set_xticklabels(station_names, rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/plots/station_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Correlation analysis
    print("4. Creating correlation analysis...")
    
    # Prepare data for correlation analysis
    # Resample weather data to hourly to match power data
    df_temp_hourly = df_temp.set_index('Time').resample('H').mean()
    df_irr_hourly = df_irr.set_index('Time').resample('H').mean()
    
    # Get power data for same period
    df_power_period = df_power[(df_power['Time'] >= df_temp_hourly.index.min()) & 
                              (df_power['Time'] <= df_temp_hourly.index.max())]
    df_power_hourly = df_power_period.set_index('Time').resample('H').mean()
    
    # Merge data
    correlation_data = pd.concat([
        df_power_hourly[['power(W)']],
        df_temp_hourly[['Temp (Degree Celsius)']],
        df_irr_hourly[['Irradiance (W/m2)']]
    ], axis=1).dropna()
    
    # Create correlation matrix
    corr_matrix = correlation_data.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.3f')
    plt.title('Correlation Matrix: Power vs Weather Variables')
    plt.tight_layout()
    plt.savefig('/home/ubuntu/plots/correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations created successfully!")
    print("Files saved in /home/ubuntu/plots/:")
    print("- power_generation_analysis.png")
    print("- weather_analysis.png") 
    print("- station_comparison.png")
    print("- correlation_analysis.png")
    
    return correlation_data

if __name__ == "__main__":
    correlation_data = create_initial_visualizations()
    print(f"\nCorrelation analysis summary:")
    print(correlation_data.corr())

