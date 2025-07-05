# Solar Power Prediction Model

A machine learning project for predicting solar panel power generation using weather data and historical patterns.

## 🎯 Project Overview

This project develops a highly accurate machine learning model for predicting solar power generation with **99.8% accuracy (R² = 0.998)** using Random Forest algorithm.

### Key Results
- **Best Model**: Random Forest with RMSE of 119.16 W
- **Dataset**: 60,030 samples from 60+ solar stations
- **Features**: 28 engineered features including weather, time, and lag variables
- **Time Period**: 3 years of hourly data (2021-2023)

## 🚀 Quick Start on AWS SageMaker

### 1. Clone Repository
```bash
git clone https://github.com/YOUR_USERNAME/solar-power-prediction.git
cd solar-power-prediction
```

### 2. Launch SageMaker Notebook Instance
1. Open AWS SageMaker Console
2. Create new Notebook Instance (ml.t3.medium recommended)
3. Upload this repository or clone directly in SageMaker

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Notebooks
Start with `notebooks/01_data_exploration.ipynb` and follow the sequence.

## 📁 Project Structure

```
solar-power-prediction/
├── notebooks/              # Jupyter notebooks for SageMaker
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/                    # Source code modules
│   ├── data_exploration.py
│   ├── data_preprocessing_efficient.py
│   ├── complete_models.py
│   └── model_evaluation.py
├── data/                   # Processed datasets
│   ├── processed_solar_sample.csv
│   ├── feature_info_sample.json
│   └── model_results.csv
├── models/                 # Trained models
│   ├── model_random_forest.pkl
│   ├── model_gradient_boosting.pkl
│   └── scaler.pkl
├── plots/                  # Generated visualizations
├── docs/                   # Documentation
│   ├── Solar_Power_Prediction_Report.md
│   └── Project_Deliverables_Summary.md
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🔧 Requirements

### Python Dependencies
- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- scipy >= 1.9.0
- joblib >= 1.1.0

### AWS SageMaker Setup
- Instance Type: ml.t3.medium or higher
- Python 3.8+ kernel
- Minimum 10GB storage

## 📊 Model Performance

| Model | RMSE | R² Score | MAE | Status |
|-------|------|----------|-----|--------|
| Random Forest | 119.16 | 0.998 | 33.28 | **RECOMMENDED** |
| Gradient Boosting | 463.74 | 0.971 | 215.48 | Good alternative |
| Linear Regression | 1049.57 | 0.854 | 546.32 | Baseline |

## 🎯 Key Features

### Top Predictive Features
1. **Power_lag_1** (67.9%) - Previous hour power output
2. **Irradiance** (17.6%) - Solar irradiance measurement  
3. **Power_Density** (12.5%) - Power per unit irradiance
4. **SolarElevation** (0.8%) - Solar elevation angle

### Feature Categories
- **Weather Features**: Temperature, Irradiance, Humidity, Wind, Rainfall
- **Time Features**: Hour, Day, Month, Season, Solar position
- **Lag Features**: Historical power patterns
- **Engineered Features**: Power density, efficiency metrics

## 🔄 Usage

### For Prediction
```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('models/model_random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare your features (28 features required)
# See feature_info_sample.json for feature list
features_scaled = scaler.transform(your_features)

# Make prediction
prediction = model.predict(features_scaled)
```

### For Training New Models
Run the notebooks in sequence:
1. Data Exploration
2. Data Preprocessing  
3. Model Training
4. Model Evaluation

## 📈 Applications

- **Grid Management**: Real-time power generation forecasting
- **Energy Trading**: Prediction intervals for risk assessment
- **Maintenance Planning**: Performance monitoring and anomaly detection
- **Capacity Planning**: Long-term infrastructure decisions

## 📚 Documentation

- [Complete Project Report](docs/Solar_Power_Prediction_Report.md)
- [Deliverables Summary](docs/Project_Deliverables_Summary.md)
- [Model Performance Analysis](plots/)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions or issues:
1. Check the documentation in `docs/`
2. Review the Jupyter notebooks for examples
3. Open an issue on GitHub

---

**Ready for Production Deployment** ✅

Built with ❤️ for renewable energy forecasting

