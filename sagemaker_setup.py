#!/usr/bin/env python3
"""
SageMaker Setup Script for Solar Power Prediction Project

This script automates the setup process for running the solar power prediction
project on AWS SageMaker. It handles dependency installation, data preparation,
and environment configuration.

Usage:
    python sagemaker_setup.py [--quick] [--install-deps] [--prepare-data]

Options:
    --quick         Run quick setup (minimal dependencies)
    --install-deps  Install all required dependencies
    --prepare-data  Prepare sample data for demonstration
    --help         Show this help message
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors."""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} - Success")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e.stderr}")
        return None

def check_environment():
    """Check if running in SageMaker environment."""
    print("🔍 Checking environment...")
    
    # Check if running in SageMaker
    is_sagemaker = os.path.exists('/opt/ml') or 'sagemaker' in os.getcwd().lower()
    
    # Check Python version
    python_version = sys.version_info
    print(f"📍 Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check available memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"💾 Available memory: {memory.available / (1024**3):.1f} GB")
    except ImportError:
        print("💾 Memory info not available (psutil not installed)")
    
    # Check disk space
    disk_usage = os.statvfs('.')
    free_space = disk_usage.f_frsize * disk_usage.f_bavail / (1024**3)
    print(f"💿 Free disk space: {free_space:.1f} GB")
    
    if is_sagemaker:
        print("✅ Running in SageMaker environment")
    else:
        print("⚠️  Not detected as SageMaker environment")
    
    return is_sagemaker

def install_dependencies(quick=False):
    """Install required Python packages."""
    print("📦 Installing dependencies...")
    
    if quick:
        # Minimal dependencies for quick start
        packages = [
            "pandas>=1.5.0",
            "numpy>=1.21.0", 
            "scikit-learn>=1.1.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0"
        ]
    else:
        # Full dependencies from requirements.txt
        if os.path.exists('requirements.txt'):
            run_command("pip install -r requirements.txt", 
                       "Installing from requirements.txt")
            return
        else:
            packages = [
                "pandas>=1.5.0",
                "numpy>=1.21.0",
                "scikit-learn>=1.1.0", 
                "matplotlib>=3.5.0",
                "seaborn>=0.11.0",
                "scipy>=1.9.0",
                "joblib>=1.1.0",
                "jupyter>=1.0.0",
                "ipykernel>=6.0.0"
            ]
    
    # Install packages
    for package in packages:
        run_command(f"pip install '{package}'", f"Installing {package}")
    
    # Verify installation
    print("🔍 Verifying installation...")
    try:
        import pandas, numpy, sklearn, matplotlib, seaborn
        print("✅ All core packages installed successfully")
    except ImportError as e:
        print(f"❌ Package verification failed: {e}")

def prepare_sample_data():
    """Create sample data for demonstration if original data not available."""
    print("📊 Preparing sample data...")
    
    # Check if processed data already exists
    if os.path.exists('data/processed_solar_sample.csv'):
        print("✅ Sample data already exists")
        return
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Generate sample data
    try:
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        print("🔄 Generating sample solar power data...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate time series (1 month of hourly data)
        start_date = datetime(2023, 6, 1)  # Summer month for good solar data
        dates = [start_date + timedelta(hours=i) for i in range(24*30)]
        
        n_samples = len(dates)
        
        # Generate realistic solar patterns
        hours = np.array([d.hour for d in dates])
        days = np.array([d.timetuple().tm_yday for d in dates])
        
        # Solar irradiance with daily pattern
        irradiance = np.maximum(0, 
            800 * np.sin(np.pi * (hours - 6) / 12) * 
            (1 + 0.1 * np.sin(2 * np.pi * days / 365)) + 
            np.random.normal(0, 100, n_samples)
        )
        
        # Temperature with daily and seasonal variation
        temperature = (20 + 8 * np.sin(2 * np.pi * days / 365) + 
                      5 * np.sin(np.pi * (hours - 6) / 12) + 
                      np.random.normal(0, 2, n_samples))
        
        # Power generation based on irradiance and temperature
        power = np.maximum(0, 
            irradiance * 2.5 + temperature * 20 + 
            np.random.normal(0, 150, n_samples)
        )
        
        # Create lag features
        power_lag_1 = np.roll(power, 1)
        power_lag_1[0] = power[0]  # Fill first value
        
        # Create DataFrame
        sample_data = pd.DataFrame({
            'Time': dates,
            'Power_W': power,
            'Irradiance': irradiance,
            'Temperature': temperature,
            'RelativeHumidity': np.random.normal(60, 15, n_samples),
            'WindSpeed': np.random.exponential(5, n_samples),
            'Hour': hours,
            'Power_lag_1': power_lag_1,
            'Power_Density': power / (irradiance + 1e-6),
            'SolarElevation': np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
        })
        
        # Save sample data
        sample_data.to_csv('data/processed_solar_sample.csv', index=False)
        
        # Create feature info
        feature_info = {
            'target': 'Power_W',
            'features': [col for col in sample_data.columns if col not in ['Time', 'Power_W']],
            'n_samples': len(sample_data),
            'n_features': len(sample_data.columns) - 2
        }
        
        with open('data/feature_info_sample.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print(f"✅ Generated {n_samples} samples of solar power data")
        print(f"📁 Saved to: data/processed_solar_sample.csv")
        
    except Exception as e:
        print(f"❌ Failed to generate sample data: {e}")

def setup_jupyter_kernel():
    """Set up Jupyter kernel for the project."""
    print("🔧 Setting up Jupyter kernel...")
    
    kernel_name = "solar-power-prediction"
    
    # Install ipykernel if not available
    run_command("pip install ipykernel", "Installing ipykernel")
    
    # Create kernel
    run_command(f"python -m ipykernel install --user --name {kernel_name} --display-name 'Solar Power Prediction'",
               "Creating Jupyter kernel")

def verify_setup():
    """Verify that the setup is complete and working."""
    print("🔍 Verifying setup...")
    
    checks = []
    
    # Check Python packages
    try:
        import pandas, numpy, sklearn, matplotlib, seaborn, joblib
        checks.append(("Python packages", True))
    except ImportError:
        checks.append(("Python packages", False))
    
    # Check data files
    data_exists = os.path.exists('data/processed_solar_sample.csv')
    checks.append(("Sample data", data_exists))
    
    # Check notebooks
    notebooks_exist = all(os.path.exists(f'notebooks/{nb}') for nb in [
        '01_data_exploration.ipynb',
        '02_data_preprocessing.ipynb', 
        '03_model_training.ipynb',
        '04_model_evaluation.ipynb'
    ])
    checks.append(("Jupyter notebooks", notebooks_exist))
    
    # Check models directory
    models_dir = os.path.exists('models')
    checks.append(("Models directory", models_dir))
    
    # Print results
    print("\n📋 Setup Verification:")
    print("-" * 40)
    for check_name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}")
    
    all_good = all(status for _, status in checks)
    
    if all_good:
        print("\n🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Open Jupyter and navigate to the notebooks/ folder")
        print("2. Start with 01_data_exploration.ipynb")
        print("3. Run notebooks in sequence: 01 → 02 → 03 → 04")
        print("4. Check the docs/ folder for detailed guides")
    else:
        print("\n⚠️  Some issues detected. Please check the failed items above.")
    
    return all_good

def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup Solar Power Prediction project for SageMaker")
    parser.add_argument('--quick', action='store_true', help='Quick setup with minimal dependencies')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies only')
    parser.add_argument('--prepare-data', action='store_true', help='Prepare sample data only')
    parser.add_argument('--verify', action='store_true', help='Verify setup only')
    
    args = parser.parse_args()
    
    print("🌞 Solar Power Prediction - SageMaker Setup")
    print("=" * 50)
    
    # Check environment
    is_sagemaker = check_environment()
    
    if args.verify:
        verify_setup()
        return
    
    if args.install_deps:
        install_dependencies(quick=args.quick)
        return
    
    if args.prepare_data:
        prepare_sample_data()
        return
    
    # Full setup
    print("\n🔄 Starting full setup...")
    
    # Install dependencies
    install_dependencies(quick=args.quick)
    
    # Prepare sample data
    prepare_sample_data()
    
    # Setup Jupyter kernel
    if is_sagemaker:
        setup_jupyter_kernel()
    
    # Verify setup
    verify_setup()
    
    print("\n📚 Documentation available in docs/ folder:")
    print("- Quick_Start_Guide.md - Get started in 15 minutes")
    print("- AWS_SageMaker_Deployment_Guide.md - Complete deployment guide")
    print("- Solar_Power_Prediction_Report.md - Technical report")

if __name__ == "__main__":
    main()

