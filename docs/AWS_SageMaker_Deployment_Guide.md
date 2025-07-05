# AWS SageMaker Deployment Guide

This guide provides step-by-step instructions for deploying and running the Solar Power Prediction project on AWS SageMaker.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setting up SageMaker](#setting-up-sagemaker)
3. [Uploading the Project](#uploading-the-project)
4. [Running the Notebooks](#running-the-notebooks)
5. [Model Deployment](#model-deployment)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### AWS Account Requirements
- Active AWS account with SageMaker access
- IAM permissions for SageMaker, S3, and EC2
- Basic familiarity with AWS console

### Local Requirements
- Git installed on your local machine
- GitHub account (for cloning the repository)
- Web browser for AWS console access

## Setting up SageMaker

### Step 1: Access SageMaker Console
1. Log into your AWS Console
2. Navigate to **Amazon SageMaker** service
3. Select your preferred region (e.g., us-east-1, us-west-2)

### Step 2: Create a Notebook Instance
1. In SageMaker console, click **Notebook instances** in the left sidebar
2. Click **Create notebook instance**
3. Configure the instance:
   - **Notebook instance name**: `solar-power-prediction`
   - **Notebook instance type**: `ml.t3.medium` (recommended for development)
   - **Platform identifier**: `notebook-al2-v2` (Amazon Linux 2)
   - **Volume size**: 20 GB (minimum recommended)

### Step 3: Configure IAM Role
1. Under **Permissions and encryption**:
   - **IAM role**: Create a new role or use existing
   - If creating new: Select **Any S3 bucket** for simplicity
   - **Root access**: Enable (for package installations)

### Step 4: Network Configuration (Optional)
- **VPC**: Default VPC is fine for most cases
- **Subnet**: Default subnet
- **Security group**: Default security group

### Step 5: Launch Instance
1. Click **Create notebook instance**
2. Wait 5-10 minutes for instance to be **InService**
3. Click **Open Jupyter** when ready

## Uploading the Project

### Method 1: Clone from GitHub (Recommended)

1. In Jupyter, open a **Terminal** (New → Terminal)
2. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/solar-power-prediction.git
cd solar-power-prediction
```

### Method 2: Upload Files Manually

1. Download the project as ZIP from GitHub
2. In Jupyter, use **Upload** button to upload the ZIP file
3. Extract in terminal:
```bash
unzip solar-power-prediction.zip
cd solar-power-prediction
```

### Method 3: Upload to S3 and Download

1. Upload project to S3 bucket
2. In SageMaker terminal:
```bash
aws s3 cp s3://your-bucket/solar-power-prediction.zip .
unzip solar-power-prediction.zip
cd solar-power-prediction
```

## Installing Dependencies

### Step 1: Install Required Packages
In the SageMaker terminal, run:
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python -c "import pandas, numpy, sklearn, matplotlib, seaborn; print('All packages installed successfully!')"
```

## Running the Notebooks

### Notebook Execution Order
Run the notebooks in the following sequence:

#### 1. Data Exploration (`01_data_exploration.ipynb`)
- **Purpose**: Understand the dataset structure and patterns
- **Runtime**: 5-10 minutes
- **Key outputs**: Data visualizations and correlation analysis

#### 2. Data Preprocessing (`02_data_preprocessing.ipynb`)
- **Purpose**: Clean data and engineer features
- **Runtime**: 10-15 minutes
- **Key outputs**: Processed dataset and feature information

#### 3. Model Training (`03_model_training.ipynb`)
- **Purpose**: Train multiple ML models
- **Runtime**: 15-30 minutes
- **Key outputs**: Trained models and performance comparison

#### 4. Model Evaluation (`04_model_evaluation.ipynb`)
- **Purpose**: Detailed model analysis and diagnostics
- **Runtime**: 10-15 minutes
- **Key outputs**: Model diagnostics and deployment recommendations

### Running Tips
1. **Kernel**: Use **Python 3** kernel for all notebooks
2. **Memory**: Monitor memory usage in top-right corner
3. **Saving**: Notebooks auto-save, but manually save important results
4. **Restart**: Restart kernel if you encounter memory issues

## Model Deployment Options

### Option 1: Real-time Endpoint (Production)

#### Create Model
```python
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create model
sklearn_model = SKLearnModel(
    model_data='s3://your-bucket/model.tar.gz',
    role=role,
    entry_point='inference.py',
    framework_version='0.23-1'
)

# Deploy endpoint
predictor = sklearn_model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium'
)
```

#### Make Predictions
```python
# Prepare input data
import numpy as np
test_data = np.array([[400, 25, 12, 2000, 5.0]])  # Example features

# Get prediction
prediction = predictor.predict(test_data)
print(f"Predicted power: {prediction[0]:.2f} W")
```

### Option 2: Batch Transform (Batch Processing)

```python
# Create transformer
transformer = sklearn_model.transformer(
    instance_count=1,
    instance_type='ml.m5.large',
    output_path='s3://your-bucket/batch-predictions/'
)

# Run batch transform
transformer.transform(
    data='s3://your-bucket/input-data.csv',
    content_type='text/csv'
)
```

### Option 3: Local Inference (Development)

```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/model_random_forest.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare features
features = pd.DataFrame({
    'Irradiance': [400],
    'Temperature': [25],
    'Hour': [12],
    'Power_lag_1': [2000],
    'Power_Density': [5.0]
    # ... add all required features
})

# Scale and predict
features_scaled = scaler.transform(features)
prediction = model.predict(features_scaled)
print(f"Predicted power: {prediction[0]:.2f} W")
```

## Performance Optimization

### Instance Types for Different Workloads

| Workload | Instance Type | vCPUs | Memory | Cost/Hour |
|----------|---------------|-------|---------|-----------|
| Development | ml.t3.medium | 2 | 4 GB | $0.05 |
| Training | ml.m5.large | 2 | 8 GB | $0.10 |
| Production | ml.m5.xlarge | 4 | 16 GB | $0.19 |
| Batch Processing | ml.c5.2xlarge | 8 | 16 GB | $0.34 |

### Memory Management
```python
# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Clear variables when done
del large_dataframe
import gc
gc.collect()
```

### Data Storage Best Practices
1. **Use S3** for large datasets (>1GB)
2. **Parquet format** for faster loading
3. **Data partitioning** for large time series
4. **Compression** to reduce storage costs

## Cost Management

### Estimated Costs (us-east-1 region)
- **Notebook Instance** (ml.t3.medium): ~$36/month if running 24/7
- **Training** (ml.m5.large): ~$0.10/hour during training
- **Inference Endpoint** (ml.t2.medium): ~$35/month if running 24/7
- **Storage** (20GB EBS): ~$2/month

### Cost Optimization Tips
1. **Stop instances** when not in use
2. **Use Spot instances** for training (up to 90% savings)
3. **Schedule automatic shutdown** for development instances
4. **Monitor usage** with AWS Cost Explorer

## Troubleshooting

### Common Issues and Solutions

#### 1. Package Installation Errors
```bash
# Update pip first
pip install --upgrade pip

# Install with specific versions
pip install pandas==1.5.0 scikit-learn==1.1.0

# Use conda if pip fails
conda install -c conda-forge package_name
```

#### 2. Memory Errors
```python
# Reduce data size
data_sample = data.sample(frac=0.1)  # Use 10% of data

# Use chunking for large files
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    process_chunk(chunk)
```

#### 3. Kernel Crashes
1. **Restart kernel**: Kernel → Restart
2. **Clear outputs**: Cell → All Output → Clear
3. **Increase instance size**: Stop instance, change type, restart

#### 4. Model Loading Errors
```python
# Check file paths
import os
print(os.listdir('models/'))

# Verify model format
import joblib
try:
    model = joblib.load('models/model_random_forest.pkl')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error: {e}")
```

#### 5. Permission Errors
```bash
# Fix file permissions
chmod +x script.py

# Check IAM role permissions in AWS console
# Ensure SageMaker execution role has necessary permissions
```

### Getting Help
1. **AWS Documentation**: [SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
2. **AWS Support**: Create support case for technical issues
3. **Community Forums**: AWS re:Post, Stack Overflow
4. **GitHub Issues**: Report project-specific issues

## Security Best Practices

### Data Protection
1. **Encrypt data** at rest and in transit
2. **Use IAM roles** instead of access keys
3. **Enable VPC** for network isolation
4. **Regular security updates** for packages

### Access Control
1. **Principle of least privilege** for IAM roles
2. **Multi-factor authentication** for AWS console
3. **Regular access reviews** for team members
4. **Audit logs** with CloudTrail

## Next Steps

After successful deployment:

1. **Monitor Performance**: Set up CloudWatch metrics
2. **Automate Retraining**: Schedule periodic model updates
3. **A/B Testing**: Compare model versions
4. **Scale Infrastructure**: Auto-scaling for production loads
5. **Integration**: Connect to existing systems via APIs

## Support and Maintenance

### Regular Tasks
- **Weekly**: Check model performance metrics
- **Monthly**: Review and retrain models with new data
- **Quarterly**: Evaluate cost optimization opportunities
- **Annually**: Review and update infrastructure

### Monitoring Setup
```python
# CloudWatch custom metrics
import boto3
cloudwatch = boto3.client('cloudwatch')

cloudwatch.put_metric_data(
    Namespace='SolarPowerPrediction',
    MetricData=[
        {
            'MetricName': 'PredictionAccuracy',
            'Value': accuracy_score,
            'Unit': 'Percent'
        }
    ]
)
```

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**Compatibility**: AWS SageMaker, Python 3.8+, scikit-learn 1.1+

For questions or issues, please refer to the project documentation or create an issue in the GitHub repository.

