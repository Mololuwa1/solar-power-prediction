import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

def load_and_prepare_data():
    """Load and prepare data"""
    
    print("Loading processed dataset...")
    data = pd.read_csv('/home/ubuntu/processed_solar_sample.csv')
    
    with open('/home/ubuntu/feature_info_sample.json', 'r') as f:
        feature_info = json.load(f)
    
    # Prepare features and target
    X = data[feature_info['features']]
    y = data[feature_info['target']]
    
    # Handle categorical features
    if 'categorical' in feature_info and feature_info['categorical']:
        for cat_col in feature_info['categorical']:
            if cat_col in X.columns:
                X = pd.get_dummies(X, columns=[cat_col], prefix=cat_col)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X_train.columns

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate a model"""
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    print(f"{model_name}: Test RMSE = {test_rmse:.2f}, Test R² = {test_r2:.3f}")
    
    return {
        'model_name': model_name,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'cv_rmse': cv_rmse
    }, y_test_pred

def main():
    """Complete model training and evaluation"""
    
    print("COMPLETING SOLAR POWER PREDICTION MODELS")
    print("="*50)
    
    # Load data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_prepare_data()
    
    # Define models to train
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=50, random_state=42)
    }
    
    results = []
    predictions = {}
    
    # Train remaining models
    for name, model in models.items():
        print(f"Training {name}...")
        metrics, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results.append(metrics)
        predictions[name] = y_pred
        
        # Save model
        joblib.dump(model, f'/home/ubuntu/model_{name.replace(" ", "_").lower()}.pkl')
    
    # Load existing models and evaluate
    try:
        lr_model = joblib.load('/home/ubuntu/model_linear_regression.pkl')
        lr_metrics, lr_pred = evaluate_model(lr_model, X_train, X_test, y_train, y_test, 'Linear Regression')
        results.append(lr_metrics)
        predictions['Linear Regression'] = lr_pred
        
        ridge_model = joblib.load('/home/ubuntu/model_ridge_regression.pkl')
        ridge_metrics, ridge_pred = evaluate_model(ridge_model, X_train, X_test, y_train, y_test, 'Ridge Regression')
        results.append(ridge_metrics)
        predictions['Ridge Regression'] = ridge_pred
    except:
        print("Could not load existing models")
    
    # Create results DataFrame
    df_results = pd.DataFrame(results)
    df_results.to_csv('/home/ubuntu/model_results.csv', index=False)
    
    # Create simple visualization
    import os
    os.makedirs('/home/ubuntu/plots', exist_ok=True)
    
    plt.figure(figsize=(12, 8))
    
    # Model comparison
    plt.subplot(2, 2, 1)
    plt.bar(df_results['model_name'], df_results['test_rmse'])
    plt.title('Test RMSE Comparison')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    
    plt.subplot(2, 2, 2)
    plt.bar(df_results['model_name'], df_results['test_r2'])
    plt.title('Test R² Comparison')
    plt.ylabel('R²')
    plt.xticks(rotation=45)
    
    # Prediction scatter plot for best model
    best_model_idx = df_results['test_rmse'].idxmin()
    best_model_name = df_results.loc[best_model_idx, 'model_name']
    best_r2 = df_results.loc[best_model_idx, 'test_r2']
    best_rmse = df_results.loc[best_model_idx, 'test_rmse']
    
    plt.subplot(2, 2, 3)
    if best_model_name in predictions:
        y_pred_best = predictions[best_model_name]
        plt.scatter(y_test, y_pred_best, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Power (W)')
        plt.ylabel('Predicted Power (W)')
        plt.title(f'Best Model: {best_model_name}\nR² = {best_r2:.3f}, RMSE = {best_rmse:.1f}')
    
    plt.tight_layout()
    plt.savefig('/home/ubuntu/plots/final_model_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save scaler
    joblib.dump(scaler, '/home/ubuntu/scaler.pkl')
    
    print("\n" + "="*50)
    print("MODEL TRAINING COMPLETE")
    print("="*50)
    print("\nModel Performance Summary:")
    for _, row in df_results.iterrows():
        print(f"{row['model_name']}: RMSE = {row['test_rmse']:.2f}, R² = {row['test_r2']:.3f}")
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best RMSE: {best_rmse:.2f}")
    print(f"Best R²: {best_r2:.3f}")
    
    print("\nFiles saved:")
    print("- Model files: /home/ubuntu/model_*.pkl")
    print("- Scaler: /home/ubuntu/scaler.pkl")
    print("- Results: /home/ubuntu/model_results.csv")
    print("- Plot: /home/ubuntu/plots/final_model_results.png")
    
    return df_results

if __name__ == "__main__":
    results = main()

