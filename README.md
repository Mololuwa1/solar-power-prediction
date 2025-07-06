# Solar Power Prediction Model

A comprehensive machine learning project for predicting both **solar power output (W)** and **energy generation (kWh)** using weather data and historical patterns.

## 🎯 Project Highlights

- **99.8% accuracy** for power prediction (Random Forest)
- **99.57% accuracy** for energy generation prediction (Gradient Boosting)
- **Production-ready models** with comprehensive testing framework
- **AWS SageMaker optimized** with Jupyter notebooks
- **Complete documentation** and deployment guides

## ⚡ Energy vs Power Prediction

### Power Prediction (W)
- Predicts instantaneous power output
- Best for real-time monitoring
- RMSE: 119.16 W, R²: 0.998

### Energy Generation Prediction (kWh) ⭐ **RECOMMENDED**
- Predicts cumulative energy generation
- Better for planning and forecasting
- RMSE: 0.0499 kWh, R²: 0.9957
- **Total Energy Error: 0.64%**

## 🚀 Quick Start on AWS SageMaker

### Option 1: Complete Setup
```bash
git clone https://github.com/Mololuwa1/solar-power-prediction.git
cd solar-power-prediction
python sagemaker_energy_setup.py --setup-type complete
```

### Option 2: Energy Generation Only
```bash
python sagemaker_energy_setup.py --setup-type energy-only
```

### Option 3: Quick Demo
```bash
python sagemaker_energy_setup.py --setup-type quick
```

## 📓 Jupyter Notebooks (SageMaker Ready)

| Notebook | Purpose | Key Features |
|----------|---------|--------------|
| `00_quick_start.ipynb` | Setup verification | Environment check, sample data |
| `01_data_exploration.ipynb` | Data analysis | 124 files, 60+ stations analyzed |
| `02_data_preprocessing.ipynb` | Feature engineering | 28 features, time series processing |
| `03_model_training.ipynb` | Power prediction | Random Forest, 99.8% accuracy |
| `04_model_evaluation.ipynb` | Model diagnostics | Performance analysis, visualizations |
| `05_energy_generation_training.ipynb` | **Energy prediction** | **99.57% accuracy, production-ready** |
| `06_energy_generation_testing.ipynb` | **Testing framework** | **Test on new datasets** |

## 📊 Model Performance

### Energy Generation Models (Recommended)
| Model | RMSE (kWh) | R² | Total Energy Error |
|-------|------------|----|--------------------|
| **Gradient Boosting** | **0.0499** | **99.57%** | **0.64%** |
| Random Forest | 0.0523 | 99.52% | 0.71% |
| Ridge Regression | 0.1247 | 97.89% | 1.89% |
| Linear Regression | 0.1389 | 96.45% | 2.15% |

### Power Prediction Models
| Model | RMSE (W) | R² | Performance |
|-------|----------|----|-----------| 
| **Random Forest** | **119.16** | **99.8%** | **Excellent** |
| Gradient Boosting | 463.74 | 97.1% | Very Good |
| Linear Regression | 1049.57 | 85.4% | Good |

## 🎯 Use Cases

### Energy Generation Prediction (Primary)
- ✅ **Grid management and planning**
- ✅ **Financial forecasting and revenue optimization**
- ✅ **Energy storage optimization**
- ✅ **Maintenance scheduling**
- ✅ **Daily/weekly energy planning**

### Power Prediction (Secondary)
- ✅ **Real-time monitoring**
- ✅ **Instantaneous load balancing**
- ✅ **Performance diagnostics**

## 📁 Project Structure

```
solar-power-prediction/
├── 📓 notebooks/              # Jupyter notebooks for SageMaker
│   ├── 00_quick_start.ipynb   # Setup verification
│   ├── 05_energy_generation_training.ipynb  # Energy model training
│   └── 06_energy_generation_testing.ipynb   # Testing framework
├── 🐍 src/                    # Python source files
├── 📊 data/                   # Datasets and sample data
├── 🤖 models/                 # Power prediction models
├── ⚡ energy_models/          # Energy generation models
├── 📈 plots/                  # Visualizations
├── 📚 docs/                   # Documentation
├── ⚙️ sagemaker_energy_setup.py  # SageMaker setup script
└── 📋 requirements.txt        # Dependencies
```

## 🔧 Testing on New Datasets

### Quick Testing
```python
# Test energy generation model
python test_energy_generation.py --data_path "your_data.csv"

# Test all models
python test_energy_generation.py --data_path "your_data.csv" --model_type all
```

### Required Data Format
| Column | Description | Required |
|--------|-------------|----------|
| `Time` | Timestamp | ✅ |
| `generation(kWh)` | Energy generation | ✅ |
| `Irradiance` | Solar irradiance (W/m²) | ✅ |
| `Temperature` | Temperature (°C) | ✅ |
| `RelativeHumidity` | Humidity (%) | ⚠️ |
| `WindSpeed` | Wind speed (m/s) | ⚠️ |

## 📈 Performance Interpretation

| Total Energy Error | Assessment | Action |
|-------------------|------------|--------|
| < 5% | ✅ **Excellent** | Ready for production |
| 5-10% | ⚠️ **Good** | Monitor performance |
| 10-20% | ⚠️ **Acceptable** | Consider retraining |
| > 20% | ❌ **Poor** | Retraining required |

## 🌟 Key Features

- **Dual Prediction Models**: Both power (W) and energy (kWh) prediction
- **Time Series Features**: Lag features, rolling statistics, seasonal patterns
- **Weather Integration**: Comprehensive weather data processing
- **Production Ready**: Saved models with preprocessing pipelines
- **Comprehensive Testing**: Framework for validating on new datasets
- **SageMaker Optimized**: Native Jupyter notebook support
- **Automated Setup**: One-command environment setup
- **Cost Optimized**: Includes cost management best practices

## 📚 Documentation

- **[Quick Start Guide](docs/Quick_Start_Guide.md)** - Get running in 15 minutes
- **[AWS SageMaker Deployment Guide](docs/AWS_SageMaker_Deployment_Guide.md)** - Complete setup instructions
- **[Energy Generation Prediction Guide](Energy_Generation_Prediction_Guide.md)** - Detailed energy modeling guide
- **[Testing New Datasets Guide](Testing_New_Datasets_Guide.md)** - How to test on your data

## 🚀 Deployment Options

### AWS SageMaker (Recommended)
- Native Jupyter notebook support
- Scalable compute instances
- Integrated with AWS ecosystem
- Cost-effective with auto-shutdown

### Local Development
```bash
pip install -r requirements.txt
jupyter notebook notebooks/00_quick_start.ipynb
```

### Google Colab
Upload notebooks to Colab and run with GPU acceleration.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Solar panel data from multiple stations and weather sources
- Built with scikit-learn, pandas, and matplotlib
- Optimized for AWS SageMaker deployment

---

**🌞 Ready to predict solar energy generation with 99.57% accuracy!**

