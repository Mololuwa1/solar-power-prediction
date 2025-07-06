#!/usr/bin/env python3
"""
Test Energy Generation Prediction Model on New Datasets

This script allows you to test the trained energy generation prediction model
on new datasets. It handles data preprocessing, feature engineering,
and provides comprehensive evaluation metrics for energy forecasting.

Usage:
    python test_energy_generation.py --data_path "path/to/new_data.csv" --model_type "gradient_boosting"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys

warnings.filterwarnings('ignore')

class EnergyGenerationTester:
    """Class to test energy generation prediction models on new datasets."""
    
    def __init__(self, models_dir="energy_models", plots_dir="energy_test_results"):
        """
        Initialize the energy generation tester.
        
        Args:
            models_dir (str): Directory containing trained energy models
            plots_dir (str): Directory to save test results
        """
        self.models_dir = Path(models_dir)
        self.plots_dir = Path(plots_dir)
        self.plots_dir.mkdir(exist_ok=True)
        
        # Load trained models and scaler
        self.models = {}
        self.scaler = None
        self.feature_info = None
        
        self._load_models()
        
    def _load_models(self):
        """Load all available trained energy models and preprocessing components."""
        print("🔄 Loading trained energy generation models...")
        
        # Load scaler
        scaler_path = self.models_dir / "energy_scaler.pkl"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print("✅ Energy scaler loaded")
        else:
            print("❌ Energy scaler not found")
            
        # Load feature information
        feature_info_path = self.models_dir / "energy_feature_info.json"
        if feature_info_path.exists():
            with open(feature_info_path, 'r') as f:
                self.feature_info = json.load(f)
            print("✅ Energy feature info loaded")
        
        # Load models
        model_files = {
            'random_forest': 'energy_model_random_forest.pkl',
            'gradient_boosting': 'energy_model_gradient_boosting.pkl',
            'linear_regression': 'energy_model_linear_regression.pkl',
            'ridge_regression': 'energy_model_ridge_regression.pkl'
        }
        
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
                print(f"✅ {model_name} energy model loaded")
            else:
                print(f"⚠️  {model_name} energy model not found")
                
        if not self.models:
            raise FileNotFoundError("No trained energy models found! Please ensure models are in the energy_models/ directory.")
    
    def load_new_dataset(self, data_path, target_column='generation(kWh)', time_column='Time'):
        """
        Load and validate new dataset for energy generation testing.
        
        Args:
            data_path (str): Path to the new dataset
            target_column (str): Name of the target variable column
            time_column (str): Name of the time column
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print(f"📊 Loading new dataset for energy generation testing: {data_path}")
        
        # Load data
        if data_path.endswith('.csv'):
            data = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'):
            data = pd.read_excel(data_path)
        else:
            raise ValueError("Unsupported file format. Please use CSV or Excel files.")
        
        print(f"✅ Dataset loaded: {data.shape}")
        print(f"📋 Columns: {list(data.columns)}")
        
        # Convert time column if exists
        if time_column in data.columns:
            try:
                data[time_column] = pd.to_datetime(data[time_column])
                data = data.sort_values(time_column).reset_index(drop=True)
                print(f"✅ Time column converted and sorted")
            except:
                print(f"⚠️  Could not convert '{time_column}' to datetime")
        
        # Handle energy generation target
        if target_column not in data.columns:
            # Try to find alternative column names
            possible_names = ['generation(kWh)', 'energy_kwh', 'energy_generation', 'kwh', 'generation']
            found_target = None
            
            for name in possible_names:
                if name in data.columns:
                    found_target = name
                    break
            
            if found_target:
                data[target_column] = data[found_target]
                print(f"✅ Using '{found_target}' as target variable")
            elif 'power(W)' in data.columns or 'Power_W' in data.columns:
                # Convert power to energy
                power_col = 'power(W)' if 'power(W)' in data.columns else 'Power_W'
                
                # Calculate time intervals
                if len(data) > 1 and time_column in data.columns:
                    time_diff = (data[time_column].iloc[1] - data[time_column].iloc[0]).total_seconds() / 3600
                else:
                    time_diff = 1.0  # Default to 1 hour
                
                # Energy = Power * Time / 1000 (convert W to kW)
                data[target_column] = data[power_col] * time_diff / 1000
                print(f"✅ Converted power to energy generation (time interval: {time_diff:.2f} hours)")
            else:
                print(f"⚠️  Target column '{target_column}' not found and cannot be derived")
        
        return data
    
    def create_energy_features(self, data, target_col='generation(kWh)', time_col='Time'):
        """Create energy-specific features for the new dataset."""
        print("🔧 Creating energy-specific features for new dataset...")
        
        df = data.copy()
        
        # Time-based features
        if time_col in df.columns:
            df['Hour'] = df[time_col].dt.hour
            df['Day'] = df[time_col].dt.day
            df['Month'] = df[time_col].dt.month
            df['DayOfWeek'] = df[time_col].dt.dayofweek
            df['DayOfYear'] = df[time_col].dt.dayofyear
            df['Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
            
            # Cyclical encoding
            df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
            df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
            df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
            
            # Solar position
            df['SolarElevation'] = np.maximum(0, 
                np.sin(np.pi * (df['Hour'] - 6) / 12) * 
                (1 + 0.1 * np.sin(2 * np.pi * df['DayOfYear'] / 365))
            )
            
            # Season
            df['Season'] = df['Month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Autumn', 10: 'Autumn', 11: 'Autumn'
            })
            
            # Peak generation hours
            df['Peak_Hours'] = ((df['Hour'] >= 10) & (df['Hour'] <= 14)).astype(int)
        
        # Energy-specific lag features
        if target_col in df.columns:
            df[f'{target_col}_lag_1'] = df[target_col].shift(1)
            df[f'{target_col}_lag_24'] = df[target_col].shift(24)
            df[f'{target_col}_lag_168'] = df[target_col].shift(168)
            
            # Rolling statistics
            df[f'{target_col}_rolling_mean_6'] = df[target_col].rolling(window=6, min_periods=1).mean()
            df[f'{target_col}_rolling_mean_24'] = df[target_col].rolling(window=24, min_periods=1).mean()
            df[f'{target_col}_rolling_std_6'] = df[target_col].rolling(window=6, min_periods=1).std()
            
            # Daily cumulative energy
            if time_col in df.columns:
                df['Daily_Cumulative_Energy'] = df.groupby(df[time_col].dt.date)[target_col].cumsum()
            
            # Energy efficiency
            if 'Irradiance' in df.columns:
                df['Energy_Efficiency'] = df[target_col] / (df['Irradiance'] / 1000 + 1e-6)
        
        # Weather-based features
        if 'Temperature' in df.columns:
            df['Temp_Efficiency_Factor'] = 1 - 0.004 * (df['Temperature'] - 25)
        
        if 'Irradiance' in df.columns:
            df['Irradiance_kW'] = df['Irradiance'] / 1000
            if 'SolarElevation' in df.columns:
                df['Clear_Sky_Index'] = df['Irradiance'] / (1000 * df['SolarElevation'] + 1e-6)
        
        if 'Rainfall' in df.columns:
            df['Rain_Impact'] = (df['Rainfall'] > 0.1).astype(int)
        
        if 'WindSpeed' in df.columns:
            df['Wind_Cooling_Effect'] = np.minimum(df['WindSpeed'] * 0.1, 1.0)
        
        print("✅ Energy-specific features created")
        return df
    
    def prepare_features(self, df, target_col='generation(kWh)'):
        """Prepare features for energy generation prediction."""
        print("🔄 Preparing features for energy generation prediction...")
        
        # Create features
        df = self.create_energy_features(df, target_col)
        
        # Get expected features from training
        if self.feature_info:
            expected_features = self.feature_info['features']
        else:
            # Fallback feature list
            expected_features = [
                'Irradiance', 'Temperature', 'RelativeHumidity', 'WindSpeed', 'Rainfall', 'SeaLevelPressure',
                'Hour', 'Day', 'Month', 'DayOfWeek', 'DayOfYear', 'Weekend',
                'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'DayOfYear_sin', 'DayOfYear_cos',
                'SolarElevation', 'Peak_Hours',
                f'{target_col}_lag_1', f'{target_col}_lag_24', f'{target_col}_lag_168',
                f'{target_col}_rolling_mean_6', f'{target_col}_rolling_mean_24', f'{target_col}_rolling_std_6',
                'Daily_Cumulative_Energy', 'Energy_Efficiency',
                'Temp_Efficiency_Factor', 'Irradiance_kW', 'Clear_Sky_Index',
                'Rain_Impact', 'Wind_Cooling_Effect',
                'Season_Autumn', 'Season_Spring', 'Season_Summer', 'Season_Winter'
            ]
        
        # Handle categorical variables
        if 'Season' in df.columns:
            season_dummies = pd.get_dummies(df['Season'], prefix='Season')
            df = pd.concat([df, season_dummies], axis=1)
        
        # Select available features
        available_features = [col for col in expected_features if col in df.columns]
        missing_features = [col for col in expected_features if col not in df.columns]
        
        if missing_features:
            print(f"⚠️  Missing features: {missing_features[:10]}...")  # Show first 10
            print("Creating placeholder features with zeros...")
            for feature in missing_features:
                df[feature] = 0
            available_features = expected_features
        
        # Prepare feature matrix
        X = df[available_features]
        
        # Prepare target
        if target_col in df.columns:
            y = df[target_col]
        else:
            print(f"⚠️  Target column '{target_col}' not found. Cannot evaluate predictions.")
            y = None
        
        # Handle missing values more thoroughly
        X = X.fillna(X.mean())
        
        # Additional NaN handling for edge cases
        X = X.replace([np.inf, -np.inf], 0)
        
        # Fill any remaining NaN with 0
        X = X.fillna(0)
        
        if y is not None:
            y = y.fillna(y.mean())
            y = y.replace([np.inf, -np.inf], 0)
            y = y.fillna(0)
        
        print(f"✅ Features prepared: {X.shape}")
        return X, y, available_features
    
    def test_energy_model(self, X, y, model_name='gradient_boosting'):
        """Test energy generation model on new data."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not available. Available models: {list(self.models.keys())}")
        
        print(f"🔄 Testing {model_name} energy generation model...")
        
        model = self.models[model_name]
        
        # Scale features if needed
        if model_name in ['linear_regression', 'ridge_regression'] and self.scaler:
            X_scaled = self.scaler.transform(X)
            y_pred = model.predict(X_scaled)
        else:
            y_pred = model.predict(X)
        
        # Calculate metrics
        results = {
            'model_name': model_name,
            'predictions': y_pred,
            'n_samples': len(y_pred)
        }
        
        if y is not None:
            # Remove NaN values
            mask = ~(np.isnan(y) | np.isnan(y_pred))
            y_clean = y[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_clean) > 0:
                rmse = np.sqrt(mean_squared_error(y_clean, y_pred_clean))
                mae = mean_absolute_error(y_clean, y_pred_clean)
                r2 = r2_score(y_clean, y_pred_clean)
                
                # Energy-specific metrics
                mape = np.mean(np.abs((y_clean - y_pred_clean) / (y_clean + 1e-6))) * 100
                total_actual = y_clean.sum()
                total_predicted = y_pred_clean.sum()
                total_error_pct = abs(total_actual - total_predicted) / total_actual * 100
                
                # Daily energy accuracy
                if len(y_clean) >= 24:
                    daily_actual = y_clean.groupby(y_clean.index // 24).sum()
                    daily_predicted = pd.Series(y_pred_clean).groupby(pd.Series(y_pred_clean).index // 24).sum()
                    daily_rmse = np.sqrt(mean_squared_error(daily_actual, daily_predicted))
                    daily_r2 = r2_score(daily_actual, daily_predicted)
                else:
                    daily_rmse = rmse
                    daily_r2 = r2
                
                results.update({
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'total_error_pct': total_error_pct,
                    'daily_rmse': daily_rmse,
                    'daily_r2': daily_r2,
                    'actual': y_clean,
                    'predicted': y_pred_clean
                })
                
                print(f"✅ {model_name} Energy Generation Results:")
                print(f"   RMSE: {rmse:.4f} kWh")
                print(f"   MAE: {mae:.4f} kWh")
                print(f"   R²: {r2:.4f}")
                print(f"   Total Energy Error: {total_error_pct:.2f}%")
                print(f"   Daily RMSE: {daily_rmse:.4f} kWh")
                print(f"   Daily R²: {daily_r2:.4f}")
            else:
                print("⚠️  No valid data points for metric calculation")
        
        return results
    
    def create_energy_evaluation_plots(self, results, save_plots=True):
        """Create evaluation plots for energy generation testing."""
        print("📊 Creating energy generation evaluation plots...")
        
        results_with_metrics = {k: v for k, v in results.items() 
                              if 'rmse' in v and 'actual' in v}
        
        if not results_with_metrics:
            print("⚠️  No results with metrics available for plotting")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Model comparison - RMSE
        model_names = list(results_with_metrics.keys())
        rmse_values = [results_with_metrics[name]['rmse'] for name in model_names]
        
        axes[0, 0].bar(model_names, rmse_values, color='skyblue')
        axes[0, 0].set_title('Energy Model Comparison - RMSE (kWh)')
        axes[0, 0].set_ylabel('RMSE (kWh)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Model comparison - Total Energy Error
        total_error_values = [results_with_metrics[name]['total_error_pct'] for name in model_names]
        
        axes[0, 1].bar(model_names, total_error_values, color='lightcoral')
        axes[0, 1].set_title('Total Energy Prediction Error (%)')
        axes[0, 1].set_ylabel('Error (%)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Daily accuracy comparison
        daily_r2_values = [results_with_metrics[name]['daily_r2'] for name in model_names]
        
        axes[0, 2].bar(model_names, daily_r2_values, color='lightgreen')
        axes[0, 2].set_title('Daily Energy Prediction R²')
        axes[0, 2].set_ylabel('Daily R² Score')
        axes[0, 2].tick_params(axis='x', rotation=45)
        axes[0, 2].grid(True, alpha=0.3)
        
        # Best model detailed analysis
        best_model = min(results_with_metrics.keys(), 
                        key=lambda x: results_with_metrics[x]['rmse'])
        best_results = results_with_metrics[best_model]
        
        # Actual vs Predicted
        axes[1, 0].scatter(best_results['actual'], best_results['predicted'], alpha=0.6)
        axes[1, 0].plot([best_results['actual'].min(), best_results['actual'].max()],
                       [best_results['actual'].min(), best_results['actual'].max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Energy Generation (kWh)')
        axes[1, 0].set_ylabel('Predicted Energy Generation (kWh)')
        axes[1, 0].set_title(f'Actual vs Predicted - {best_model}\\nR² = {best_results["r2"]:.4f}')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = best_results['actual'] - best_results['predicted']
        axes[1, 1].scatter(best_results['predicted'], residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Energy Generation (kWh)')
        axes[1, 1].set_ylabel('Residuals (kWh)')
        axes[1, 1].set_title(f'Residuals Plot - {best_model}')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Time series comparison
        if len(best_results['actual']) > 100:
            sample_indices = np.linspace(0, len(best_results['actual'])-1, 100, dtype=int)
            axes[1, 2].plot(sample_indices, best_results['actual'].iloc[sample_indices], 
                           'b-', label='Actual', alpha=0.7)
            axes[1, 2].plot(sample_indices, best_results['predicted'][sample_indices], 
                           'r--', label='Predicted', alpha=0.7)
        else:
            axes[1, 2].plot(best_results['actual'].values, 'b-', label='Actual', alpha=0.7)
            axes[1, 2].plot(best_results['predicted'], 'r--', label='Predicted', alpha=0.7)
        
        axes[1, 2].set_xlabel('Time Index')
        axes[1, 2].set_ylabel('Energy Generation (kWh)')
        axes[1, 2].set_title(f'Time Series - {best_model}')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.plots_dir / f"energy_model_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"📊 Energy test plots saved to: {plot_path}")
        
        plt.show()

def main():
    """Main function for testing energy generation models."""
    parser = argparse.ArgumentParser(description="Test Energy Generation Prediction Model")
    parser.add_argument('--data_path', required=True, help='Path to the new dataset')
    parser.add_argument('--model_type', default='gradient_boosting', 
                       choices=['random_forest', 'gradient_boosting', 'linear_regression', 'ridge_regression', 'all'],
                       help='Model to test')
    parser.add_argument('--target_column', default='generation(kWh)', help='Target variable column name')
    parser.add_argument('--time_column', default='Time', help='Time column name')
    
    args = parser.parse_args()
    
    try:
        # Initialize tester
        tester = EnergyGenerationTester()
        
        # Load dataset
        data = tester.load_new_dataset(args.data_path, args.target_column, args.time_column)
        
        # Prepare features
        X, y, feature_names = tester.prepare_features(data, args.target_column)
        
        # Test models
        if args.model_type == 'all':
            results = {}
            for model_name in tester.models.keys():
                try:
                    results[model_name] = tester.test_energy_model(X, y, model_name)
                except Exception as e:
                    print(f"❌ Error testing {model_name}: {e}")
        else:
            results = {args.model_type: tester.test_energy_model(X, y, args.model_type)}
        
        # Create evaluation plots
        tester.create_energy_evaluation_plots(results)
        
        print("\n" + "="*60)
        print("🎉 ENERGY GENERATION MODEL TESTING COMPLETED!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

