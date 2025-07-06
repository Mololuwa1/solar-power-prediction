#!/usr/bin/env python3
"""
Solar Energy Generation Prediction Model

This script modifies the original power prediction model to predict
energy generation (kWh) instead of instantaneous power (W).

Key differences:
- Target variable: generation(kWh) instead of power(W)
- Time-based aggregation for energy calculation
- Cumulative and interval-based predictions
- Different evaluation metrics suitable for energy forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

class SolarEnergyPredictor:
    """Class for predicting solar energy generation (kWh)."""
    
    def __init__(self, output_dir="energy_models"):
        """Initialize the energy prediction model."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def load_and_prepare_data(self, data_paths=None):
        """
        Load and prepare data for energy generation prediction.
        
        Args:
            data_paths (list): List of data file paths. If None, uses sample data.
            
        Returns:
            pd.DataFrame: Prepared dataset with energy generation targets
        """
        print("📊 Loading and preparing data for energy generation prediction...")
        
        if data_paths is None:
            # Create sample data focused on energy generation
            print("🔄 Creating sample energy generation dataset...")
            data = self._create_sample_energy_data()
        else:
            # Load actual data files
            all_data = []
            for path in data_paths:
                try:
                    df = pd.read_csv(path)
                    df['source_file'] = Path(path).stem
                    all_data.append(df)
                except Exception as e:
                    print(f"⚠️ Error loading {path}: {e}")
            
            if all_data:
                data = pd.concat(all_data, ignore_index=True)
            else:
                raise ValueError("No data could be loaded from provided paths")
        
        # Convert to energy generation format
        data = self._convert_to_energy_format(data)
        
        print(f"✅ Data prepared: {data.shape}")
        print(f"📅 Time range: {data['Time'].min()} to {data['Time'].max()}")
        print(f"⚡ Energy range: {data['generation(kWh)'].min():.3f} to {data['generation(kWh)'].max():.3f} kWh")
        
        return data
    
    def _create_sample_energy_data(self):
        """Create sample data with energy generation focus."""
        # Generate 3 months of hourly data
        start_date = datetime(2023, 6, 1)
        dates = [start_date + timedelta(hours=i) for i in range(24*90)]
        
        np.random.seed(42)
        n_samples = len(dates)
        
        # Generate realistic solar patterns
        hours = np.array([d.hour for d in dates])
        days = np.array([d.timetuple().tm_yday for d in dates])
        
        # Solar irradiance with seasonal variation
        irradiance = np.maximum(0, 
            800 * np.sin(np.pi * (hours - 6) / 12) * 
            (1 + 0.2 * np.sin(2 * np.pi * days / 365)) + 
            np.random.normal(0, 100, n_samples)
        )
        
        # Temperature with daily and seasonal patterns
        temperature = (20 + 10 * np.sin(2 * np.pi * days / 365) + 
                      6 * np.sin(np.pi * (hours - 6) / 12) + 
                      np.random.normal(0, 2, n_samples))
        
        # Power generation (W)
        power = np.maximum(0, 
            irradiance * 2.5 + temperature * 20 + 
            np.random.normal(0, 150, n_samples)
        )
        
        # Convert power to energy generation (kWh)
        # Energy = Power * Time (1 hour) / 1000 (W to kW)
        generation_kwh = power / 1000  # Convert W to kWh for 1-hour intervals
        
        # Additional weather variables
        humidity = np.random.normal(60, 15, n_samples)
        wind_speed = np.random.exponential(5, n_samples)
        rainfall = np.random.exponential(0.5, n_samples)
        pressure = np.random.normal(1013, 10, n_samples)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Time': dates,
            'generation(kWh)': generation_kwh,
            'power(W)': power,
            'Irradiance': irradiance,
            'Temperature': temperature,
            'RelativeHumidity': humidity,
            'WindSpeed': wind_speed,
            'Rainfall': rainfall,
            'SeaLevelPressure': pressure
        })
        
        return data
    
    def _convert_to_energy_format(self, data):
        """Convert power data to energy generation format."""
        print("🔄 Converting power data to energy generation format...")
        
        # Ensure Time column is datetime
        if 'Time' in data.columns:
            data['Time'] = pd.to_datetime(data['Time'])
            data = data.sort_values('Time').reset_index(drop=True)
        
        # If we have power(W) but not generation(kWh), calculate it
        if 'power(W)' in data.columns and 'generation(kWh)' not in data.columns:
            # Calculate time intervals (assuming hourly data)
            if len(data) > 1:
                time_diff = (data['Time'].iloc[1] - data['Time'].iloc[0]).total_seconds() / 3600
            else:
                time_diff = 1.0  # Default to 1 hour
            
            # Energy = Power * Time / 1000 (convert W to kW)
            data['generation(kWh)'] = data['power(W)'] * time_diff / 1000
            print(f"✅ Calculated energy generation from power data (time interval: {time_diff:.2f} hours)")
        
        # If we have generation(kWh) but not power(W), calculate it
        elif 'generation(kWh)' in data.columns and 'power(W)' not in data.columns:
            if len(data) > 1:
                time_diff = (data['Time'].iloc[1] - data['Time'].iloc[0]).total_seconds() / 3600
            else:
                time_diff = 1.0
            
            # Power = Energy * 1000 / Time
            data['power(W)'] = data['generation(kWh)'] * 1000 / time_diff
            print(f"✅ Calculated power from energy generation data")
        
        # Ensure we have the target variable
        if 'generation(kWh)' not in data.columns:
            raise ValueError("No energy generation data found. Please ensure 'generation(kWh)' column exists or provide 'power(W)' for conversion.")
        
        return data
    
    def create_energy_features(self, data):
        """Create features specifically for energy generation prediction."""
        print("🔧 Creating energy-specific features...")
        
        df = data.copy()
        
        # Time-based features
        df['Hour'] = df['Time'].dt.hour
        df['Day'] = df['Time'].dt.day
        df['Month'] = df['Time'].dt.month
        df['DayOfWeek'] = df['Time'].dt.dayofweek
        df['DayOfYear'] = df['Time'].dt.dayofyear
        df['Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Cyclical encoding for time features
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        # Solar position features
        df['SolarElevation'] = np.maximum(0, 
            np.sin(np.pi * (df['Hour'] - 6) / 12) * 
            (1 + 0.1 * np.sin(2 * np.pi * df['DayOfYear'] / 365))
        )
        
        # Energy-specific lag features
        target_col = 'generation(kWh)'
        if target_col in df.columns:
            # Previous hour, day, and week energy generation
            df[f'{target_col}_lag_1'] = df[target_col].shift(1)
            df[f'{target_col}_lag_24'] = df[target_col].shift(24)  # Same hour yesterday
            df[f'{target_col}_lag_168'] = df[target_col].shift(168)  # Same hour last week
            
            # Rolling statistics for energy
            df[f'{target_col}_rolling_mean_6'] = df[target_col].rolling(window=6, min_periods=1).mean()
            df[f'{target_col}_rolling_mean_24'] = df[target_col].rolling(window=24, min_periods=1).mean()
            df[f'{target_col}_rolling_std_6'] = df[target_col].rolling(window=6, min_periods=1).std()
            
            # Daily cumulative energy (up to current hour)
            df['Daily_Cumulative_Energy'] = df.groupby(df['Time'].dt.date)[target_col].cumsum()
            
            # Energy efficiency metrics
            if 'Irradiance' in df.columns:
                df['Energy_Efficiency'] = df[target_col] / (df['Irradiance'] / 1000 + 1e-6)  # kWh per kW irradiance
        
        # Weather-based features for energy prediction
        if 'Temperature' in df.columns:
            df['Temp_Efficiency_Factor'] = 1 - 0.004 * (df['Temperature'] - 25)  # Temperature derating
        
        if 'Irradiance' in df.columns:
            df['Irradiance_kW'] = df['Irradiance'] / 1000  # Convert to kW/m²
            df['Clear_Sky_Index'] = df['Irradiance'] / (1000 * df['SolarElevation'] + 1e-6)
        
        # Weather impact on energy
        if 'Rainfall' in df.columns:
            df['Rain_Impact'] = (df['Rainfall'] > 0.1).astype(int)  # Binary rain indicator
        
        if 'WindSpeed' in df.columns:
            df['Wind_Cooling_Effect'] = np.minimum(df['WindSpeed'] * 0.1, 1.0)  # Cooling benefit
        
        # Season classification
        df['Season'] = df['Month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
        })
        
        # Peak generation hours
        df['Peak_Hours'] = ((df['Hour'] >= 10) & (df['Hour'] <= 14)).astype(int)
        
        print("✅ Energy-specific features created")
        return df
    
    def prepare_features_and_target(self, data):
        """Prepare feature matrix and target variable for energy prediction."""
        print("🔄 Preparing features and target for energy prediction...")
        
        # Create features
        df = self.create_energy_features(data)
        
        # Define feature columns (exclude target and non-feature columns)
        exclude_cols = ['Time', 'generation(kWh)', 'power(W)', 'Season', 'source_file']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle categorical variables
        if 'Season' in df.columns:
            season_dummies = pd.get_dummies(df['Season'], prefix='Season')
            df = pd.concat([df, season_dummies], axis=1)
            feature_cols.extend(season_dummies.columns)
        
        # Select features
        X = df[feature_cols]
        y = df['generation(kWh)']
        
        # Handle missing values
        X = X.fillna(X.mean())
        y = y.fillna(y.mean())
        
        # Store feature names
        self.feature_names = feature_cols
        
        print(f"✅ Features prepared: {X.shape}")
        print(f"🎯 Target variable: generation(kWh) - {len(y)} samples")
        print(f"📊 Target range: {y.min():.3f} to {y.max():.3f} kWh")
        
        return X, y
    
    def train_energy_models(self, X, y):
        """Train multiple models for energy generation prediction."""
        print("🤖 Training energy generation prediction models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle time series
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models optimized for energy prediction
        models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=5,
                random_state=42
            ),
            'linear_regression': LinearRegression(),
            'ridge_regression': Ridge(alpha=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"🔄 Training {name}...")
            
            # Train model
            if name in ['linear_regression', 'ridge_regression']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Energy-specific metrics
            mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100
            total_actual = y_test.sum()
            total_predicted = y_pred.sum()
            total_error_pct = abs(total_actual - total_predicted) / total_actual * 100
            
            results[name] = {
                'model': model,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'total_error_pct': total_error_pct,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"✅ {name} Results:")
            print(f"   RMSE: {rmse:.4f} kWh")
            print(f"   MAE: {mae:.4f} kWh")
            print(f"   R²: {r2:.4f}")
            print(f"   MAPE: {mape:.2f}%")
            print(f"   Total Energy Error: {total_error_pct:.2f}%")
        
        # Store models
        self.models = {name: result['model'] for name, result in results.items()}
        
        return results
    
    def save_models(self):
        """Save trained models and preprocessing components."""
        print("💾 Saving energy generation models...")
        
        # Save models
        for name, model in self.models.items():
            model_path = self.output_dir / f"energy_model_{name}.pkl"
            joblib.dump(model, model_path)
            print(f"✅ Saved {name} to {model_path}")
        
        # Save scaler
        scaler_path = self.output_dir / "energy_scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Saved scaler to {scaler_path}")
        
        # Save feature information
        feature_info = {
            'target': 'generation(kWh)',
            'features': self.feature_names,
            'n_features': len(self.feature_names),
            'model_type': 'energy_generation_prediction'
        }
        
        info_path = self.output_dir / "energy_feature_info.json"
        with open(info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"✅ Saved feature info to {info_path}")
    
    def create_evaluation_plots(self, results):
        """Create evaluation plots for energy generation models."""
        print("📊 Creating energy generation evaluation plots...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Model comparison - RMSE
        model_names = list(results.keys())
        rmse_values = [results[name]['rmse'] for name in model_names]
        
        axes[0, 0].bar(model_names, rmse_values, color='skyblue')
        axes[0, 0].set_title('Model Comparison - RMSE (kWh)')
        axes[0, 0].set_ylabel('RMSE (kWh)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Model comparison - R²
        r2_values = [results[name]['r2'] for name in model_names]
        
        axes[0, 1].bar(model_names, r2_values, color='lightcoral')
        axes[0, 1].set_title('Model Comparison - R²')
        axes[0, 1].set_ylabel('R² Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Total Energy Error
        total_error_values = [results[name]['total_error_pct'] for name in model_names]
        
        axes[0, 2].bar(model_names, total_error_values, color='lightgreen')
        axes[0, 2].set_title('Total Energy Prediction Error (%)')
        axes[0, 2].set_ylabel('Error (%)')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Best model analysis
        best_model = min(results.keys(), key=lambda x: results[x]['rmse'])
        best_results = results[best_model]
        
        # Actual vs Predicted
        axes[1, 0].scatter(best_results['y_test'], best_results['y_pred'], alpha=0.6)
        axes[1, 0].plot([best_results['y_test'].min(), best_results['y_test'].max()],
                       [best_results['y_test'].min(), best_results['y_test'].max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Energy Generation (kWh)')
        axes[1, 0].set_ylabel('Predicted Energy Generation (kWh)')
        axes[1, 0].set_title(f'Actual vs Predicted - {best_model}\\nR² = {best_results["r2"]:.4f}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = best_results['y_test'] - best_results['y_pred']
        axes[1, 1].scatter(best_results['y_pred'], residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Energy Generation (kWh)')
        axes[1, 1].set_ylabel('Residuals (kWh)')
        axes[1, 1].set_title(f'Residuals Plot - {best_model}')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Time series plot (if available)
        if len(best_results['y_test']) > 50:
            sample_indices = np.linspace(0, len(best_results['y_test'])-1, 50, dtype=int)
            axes[1, 2].plot(sample_indices, best_results['y_test'].iloc[sample_indices], 
                           'b-', label='Actual', alpha=0.7)
            axes[1, 2].plot(sample_indices, best_results['y_pred'][sample_indices], 
                           'r--', label='Predicted', alpha=0.7)
        else:
            axes[1, 2].plot(best_results['y_test'].values, 'b-', label='Actual', alpha=0.7)
            axes[1, 2].plot(best_results['y_pred'], 'r--', label='Predicted', alpha=0.7)
        
        axes[1, 2].set_xlabel('Time Index')
        axes[1, 2].set_ylabel('Energy Generation (kWh)')
        axes[1, 2].set_title(f'Time Series Comparison - {best_model}')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "energy_generation_model_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Evaluation plots saved to: {plot_path}")
        
        plt.show()
        
        return best_model, best_results

def main():
    """Main function to train energy generation prediction models."""
    print("🌞 Solar Energy Generation Prediction Model Training")
    print("=" * 60)
    
    # Initialize predictor
    predictor = SolarEnergyPredictor()
    
    # Load and prepare data
    data = predictor.load_and_prepare_data()
    
    # Prepare features and target
    X, y = predictor.prepare_features_and_target(data)
    
    # Train models
    results = predictor.train_energy_models(X, y)
    
    # Create evaluation plots
    best_model, best_results = predictor.create_evaluation_plots(results)
    
    # Save models
    predictor.save_models()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎉 ENERGY GENERATION MODEL TRAINING COMPLETED!")
    print("=" * 60)
    print(f"🏆 Best Model: {best_model}")
    print(f"📊 Performance Metrics:")
    print(f"   • RMSE: {best_results['rmse']:.4f} kWh")
    print(f"   • R²: {best_results['r2']:.4f} ({best_results['r2']*100:.2f}% variance explained)")
    print(f"   • MAE: {best_results['mae']:.4f} kWh")
    print(f"   • MAPE: {best_results['mape']:.2f}%")
    print(f"   • Total Energy Error: {best_results['total_error_pct']:.2f}%")
    print(f"\n💾 Models saved to: {predictor.output_dir}/")
    print(f"📊 Evaluation plots: {predictor.output_dir}/energy_generation_model_evaluation.png")

if __name__ == "__main__":
    main()

