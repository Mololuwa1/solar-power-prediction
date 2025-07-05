import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import joblib

def load_processed_data():
    """Load the processed dataset"""
    
    print("Loading processed dataset...")
    
    # Load data
    data = pd.read_csv('/home/ubuntu/processed_solar_sample.csv')
    
    # Load feature info
    with open('/home/ubuntu/feature_info_sample.json', 'r') as f:
        feature_info = json.load(f)
    
    print(f"Dataset shape: {data.shape}")
    print(f"Features: {len(feature_info['features'])}")
    
    return data, feature_info

def prepare_model_data(data, feature_info):
    """Prepare data for modeling"""
    
    print("Preparing data for modeling...")
    
    # Separate features and target
    X = data[feature_info['features']]
    y = data[feature_info['target']]
    
    # Handle categorical features
    if 'categorical' in feature_info and feature_info['categorical']:
        for cat_col in feature_info['categorical']:
            if cat_col in X.columns:
                # One-hot encode categorical features
                X = pd.get_dummies(X, columns=[cat_col], prefix=cat_col)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame for easier handling
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a model and return metrics"""
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    metrics = {
        'model_name': model_name,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'cv_rmse': cv_rmse,
        'cv_std': np.sqrt(-cv_scores).std()
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Train RMSE: {train_rmse:.2f}")
    print(f"  Test RMSE: {test_rmse:.2f}")
    print(f"  Train R²: {train_r2:.3f}")
    print(f"  Test R²: {test_r2:.3f}")
    print(f"  CV RMSE: {cv_rmse:.2f} ± {np.sqrt(-cv_scores).std():.2f}")
    
    return metrics, y_test_pred

def train_models(X_train, X_test, y_train, y_test):
    """Train all models"""
    
    print("\n" + "="*50)
    print("TRAINING MODELS")
    print("="*50)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    results = []
    predictions = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        metrics, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(metrics)
        predictions[name] = y_pred
        
        # Save the model
        joblib.dump(model, f'/home/ubuntu/model_{name.replace(" ", "_").lower()}.pkl')
    
    return results, predictions

def create_visualizations(y_test, predictions, results):
    """Create all visualizations"""
    
    print("Creating visualizations...")
    
    # Create plots directory
    import os
    os.makedirs('/home/ubuntu/plots', exist_ok=True)
    
    # Model comparison
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # RMSE comparison
    axes[0, 0].bar(df_results['model_name'], df_results['test_rmse'])
    axes[0, 0].set_title('Test RMSE Comparison')
    axes[0, 0].set_ylabel('RMSE')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # R² comparison
    axes[0, 1].bar(df_results['model_name'], df_results['test_r2'])
    axes[0, 1].set_title('Test R² Comparison')
    axes[0, 1].set_ylabel('R²')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # MAE comparison
    axes[1, 0].bar(df_results['model_name'], df_results['test_mae'])
    axes[1, 0].set_title('Test MAE Comparison')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cross-validation RMSE
    axes[1, 1].bar(df_results['model_name'], df_results['cv_rmse'])
    axes[1, 1].set_title('Cross-Validation RMSE')
    axes[1, 1].set_ylabel('CV RMSE')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/plots/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prediction plots
    n_models = len(predictions)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (model_name, y_pred) in enumerate(predictions.items()):
        row = i // n_cols
        col = i % n_cols
        
        # Find corresponding R² score
        r2 = next(r['test_r2'] for r in results if r['model_name'] == model_name)
        rmse = next(r['test_rmse'] for r in results if r['model_name'] == model_name)
        
        axes[row, col].scatter(y_test, y_pred, alpha=0.5)
        axes[row, col].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[row, col].set_xlabel('Actual Power (W)')
        axes[row, col].set_ylabel('Predicted Power (W)')
        axes[row, col].set_title(f'{model_name}\nR² = {r2:.3f}, RMSE = {rmse:.1f}')
        axes[row, col].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(n_models, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/plots/model_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    df_results.to_csv('/home/ubuntu/model_results.csv', index=False)
    
    return df_results

def main():
    """Main function to run the complete model development pipeline"""
    
    print("SOLAR POWER PREDICTION MODEL DEVELOPMENT")
    print("="*50)
    
    # Load data
    data, feature_info = load_processed_data()
    
    # Prepare data for modeling
    X_train, X_test, y_train, y_test, scaler = prepare_model_data(data, feature_info)
    
    # Train models
    results, predictions = train_models(X_train, X_test, y_train, y_test)
    
    # Create visualizations
    df_results = create_visualizations(y_test, predictions, results)
    
    # Save scaler
    joblib.dump(scaler, '/home/ubuntu/scaler.pkl')
    
    print("\n" + "="*50)
    print("MODEL DEVELOPMENT COMPLETE")
    print("="*50)
    print("\nBest performing models:")
    best_models = df_results.nsmallest(3, 'test_rmse')
    for _, row in best_models.iterrows():
        print(f"{row['model_name']}: RMSE = {row['test_rmse']:.2f}, R² = {row['test_r2']:.3f}")
    
    print("\nFiles saved:")
    print("- Model files: /home/ubuntu/model_*.pkl")
    print("- Scaler: /home/ubuntu/scaler.pkl")
    print("- Results: /home/ubuntu/model_results.csv")
    print("- Plots: /home/ubuntu/plots/")
    
    return df_results

if __name__ == "__main__":
    results = main()

