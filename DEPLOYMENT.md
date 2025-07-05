# Deployment Instructions

## GitHub Repository Setup

### 1. Create GitHub Repository
1. Go to [GitHub](https://github.com) and sign in
2. Click **New repository** or go to https://github.com/new
3. Repository settings:
   - **Repository name**: `solar-power-prediction`
   - **Description**: `Machine Learning model for solar power prediction with 99.8% accuracy`
   - **Visibility**: Public (recommended) or Private
   - **Initialize**: Don't initialize (we have existing code)

### 2. Upload Code to GitHub

#### Option A: Command Line (Recommended)
```bash
# In your local terminal or SageMaker terminal
cd /path/to/solar-power-prediction

# Add GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/solar-power-prediction.git

# Push to GitHub
git branch -M main
git push -u origin main
```

#### Option B: GitHub Desktop
1. Download and install [GitHub Desktop](https://desktop.github.com/)
2. Clone your empty repository
3. Copy all project files to the cloned folder
4. Commit and push changes

#### Option C: Web Upload
1. Download project as ZIP file
2. Go to your GitHub repository
3. Click **uploading an existing file**
4. Drag and drop the ZIP file or select files

### 3. Verify Upload
- Check that all folders are present: `notebooks/`, `src/`, `data/`, `models/`, `docs/`, `plots/`
- Verify README.md displays correctly
- Ensure requirements.txt is included

## AWS SageMaker Deployment

### Quick Start (15 minutes)
1. **Create SageMaker Instance**:
   - Go to [AWS SageMaker Console](https://console.aws.amazon.com/sagemaker/)
   - Create notebook instance: `ml.t3.medium`, 20GB storage
   - Wait for **InService** status

2. **Clone and Setup**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/solar-power-prediction.git
   cd solar-power-prediction
   python sagemaker_setup.py --quick
   ```

3. **Run Notebooks**:
   - Open `notebooks/01_data_exploration.ipynb`
   - Run all notebooks in sequence (01 → 02 → 03 → 04)

### Production Deployment

#### Real-time Endpoint
```python
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# Create model
model = SKLearnModel(
    model_data='s3://your-bucket/model.tar.gz',
    role=sagemaker.get_execution_role(),
    entry_point='inference.py',
    framework_version='0.23-1'
)

# Deploy endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)
```

#### Batch Processing
```python
# Create transformer for batch predictions
transformer = model.transformer(
    instance_count=1,
    instance_type='ml.m5.large'
)

# Process batch data
transformer.transform('s3://your-bucket/input-data.csv')
```

## Local Development Setup

### Prerequisites
- Python 3.8+
- Git
- 8GB+ RAM recommended

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/solar-power-prediction.git
cd solar-power-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run setup
python sagemaker_setup.py --prepare-data
```

### Running Locally
```bash
# Start Jupyter
jupyter notebook

# Open notebooks in order:
# 01_data_exploration.ipynb
# 02_data_preprocessing.ipynb  
# 03_model_training.ipynb
# 04_model_evaluation.ipynb
```

## Docker Deployment

### Build Docker Image
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8080

CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t solar-power-prediction .
docker run -p 8080:8080 solar-power-prediction
```

## Cloud Platform Deployment

### Google Colab
1. Upload notebooks to Google Drive
2. Open in Colab
3. Install dependencies: `!pip install -r requirements.txt`
4. Upload data files to Colab environment

### Azure Machine Learning
1. Create Azure ML workspace
2. Upload project files
3. Create compute instance
4. Run notebooks in Azure ML Studio

### Databricks
1. Import notebooks to Databricks workspace
2. Create cluster with ML runtime
3. Install requirements via cluster libraries
4. Run notebooks on cluster

## API Deployment

### Flask API Example
```python
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load('models/model_random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = pd.DataFrame([data])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    
    return jsonify({'prediction': float(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### FastAPI Example
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

class PredictionInput(BaseModel):
    irradiance: float
    temperature: float
    hour: int
    power_lag_1: float

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Load model and make prediction
    model = joblib.load('models/model_random_forest.pkl')
    # ... prediction logic
    return {"prediction": prediction_value}
```

## Monitoring and Maintenance

### Performance Monitoring
```python
# CloudWatch metrics
import boto3
cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='SolarPowerPrediction',
    MetricData=[{
        'MetricName': 'PredictionAccuracy',
        'Value': accuracy_score,
        'Unit': 'Percent'
    }]
)
```

### Automated Retraining
```python
# Schedule retraining
import schedule
import time

def retrain_model():
    # Load new data
    # Retrain model
    # Deploy updated model
    pass

schedule.every().month.do(retrain_model)

while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour
```

## Troubleshooting

### Common Issues

**GitHub Upload Fails**:
- Check file size limits (100MB per file)
- Use Git LFS for large model files
- Verify repository permissions

**SageMaker Instance Won't Start**:
- Check AWS service limits
- Verify IAM permissions
- Try different instance type

**Package Installation Errors**:
```bash
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

**Memory Errors**:
- Use smaller data samples
- Increase instance size
- Clear variables: `del variable; gc.collect()`

### Getting Help
- **Documentation**: Check `docs/` folder
- **GitHub Issues**: Report bugs and feature requests
- **AWS Support**: For SageMaker-specific issues
- **Community**: Stack Overflow, Reddit r/MachineLearning

## Security Considerations

### Data Protection
- Encrypt sensitive data
- Use IAM roles instead of access keys
- Enable VPC for network isolation
- Regular security updates

### Access Control
- Principle of least privilege
- Multi-factor authentication
- Regular access reviews
- Audit logs with CloudTrail

## Cost Optimization

### AWS Costs
- Stop instances when not in use
- Use Spot instances for training
- Monitor with Cost Explorer
- Set up billing alerts

### Resource Management
- Right-size instances
- Use auto-scaling
- Schedule automatic shutdown
- Regular cost reviews

---

**Deployment Status**: Ready for production use ✅  
**Support**: See documentation in `docs/` folder  
**Updates**: Check GitHub releases for new versions

