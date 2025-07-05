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
    
    return metrics, y_test_pred\n\ndef train_baseline_models(X_train, X_test, y_train, y_test):\n    \"\"\"Train baseline models\"\"\"\n    \n    print(\"\\n\" + \"=\"*50)\n    print(\"TRAINING BASELINE MODELS\")\n    print(\"=\"*50)\n    \n    models = {\n        'Linear Regression': LinearRegression(),\n        'Ridge Regression': Ridge(alpha=1.0),\n        'Lasso Regression': Lasso(alpha=1.0),\n        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)\n    }\n    \n    results = []\n    predictions = {}\n    \n    for name, model in models.items():\n        metrics, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test, name)\n        results.append(metrics)\n        predictions[name] = y_pred\n        \n        # Save the model\n        joblib.dump(model, f'/home/ubuntu/model_{name.replace(\" \", \"_\").lower()}.pkl')\n    \n    return results, predictions\n\ndef train_advanced_models(X_train, X_test, y_train, y_test):\n    \"\"\"Train advanced models\"\"\"\n    \n    print(\"\\n\" + \"=\"*50)\n    print(\"TRAINING ADVANCED MODELS\")\n    print(\"=\"*50)\n    \n    models = {\n        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),\n        'Support Vector Regression': SVR(kernel='rbf', C=1.0),\n        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)\n    }\n    \n    results = []\n    predictions = {}\n    \n    for name, model in models.items():\n        print(f\"\\nTraining {name}...\")\n        metrics, y_pred = evaluate_model(model, X_train, X_test, y_train, y_test, name)\n        results.append(metrics)\n        predictions[name] = y_pred\n        \n        # Save the model\n        joblib.dump(model, f'/home/ubuntu/model_{name.replace(\" \", \"_\").lower()}.pkl')\n    \n    return results, predictions\n\ndef hyperparameter_tuning(X_train, y_train):\n    \"\"\"Perform hyperparameter tuning for best models\"\"\"\n    \n    print(\"\\n\" + \"=\"*50)\n    print(\"HYPERPARAMETER TUNING\")\n    print(\"=\"*50)\n    \n    # Random Forest tuning\n    print(\"Tuning Random Forest...\")\n    rf_params = {\n        'n_estimators': [50, 100, 200],\n        'max_depth': [10, 20, None],\n        'min_samples_split': [2, 5, 10]\n    }\n    \n    rf_grid = GridSearchCV(\n        RandomForestRegressor(random_state=42),\n        rf_params,\n        cv=3,\n        scoring='neg_mean_squared_error',\n        n_jobs=-1\n    )\n    \n    rf_grid.fit(X_train, y_train)\n    \n    print(f\"Best RF parameters: {rf_grid.best_params_}\")\n    print(f\"Best RF CV score: {np.sqrt(-rf_grid.best_score_):.2f}\")\n    \n    # Gradient Boosting tuning\n    print(\"\\nTuning Gradient Boosting...\")\n    gb_params = {\n        'n_estimators': [50, 100, 200],\n        'learning_rate': [0.05, 0.1, 0.2],\n        'max_depth': [3, 5, 7]\n    }\n    \n    gb_grid = GridSearchCV(\n        GradientBoostingRegressor(random_state=42),\n        gb_params,\n        cv=3,\n        scoring='neg_mean_squared_error',\n        n_jobs=-1\n    )\n    \n    gb_grid.fit(X_train, y_train)\n    \n    print(f\"Best GB parameters: {gb_grid.best_params_}\")\n    print(f\"Best GB CV score: {np.sqrt(-gb_grid.best_score_):.2f}\")\n    \n    # Save best models\n    joblib.dump(rf_grid.best_estimator_, '/home/ubuntu/model_random_forest_tuned.pkl')\n    joblib.dump(gb_grid.best_estimator_, '/home/ubuntu/model_gradient_boosting_tuned.pkl')\n    \n    return rf_grid.best_estimator_, gb_grid.best_estimator_\n\ndef analyze_feature_importance(model, feature_names, model_name):\n    \"\"\"Analyze feature importance\"\"\"\n    \n    if hasattr(model, 'feature_importances_'):\n        importance = model.feature_importances_\n        feature_importance = pd.DataFrame({\n            'feature': feature_names,\n            'importance': importance\n        }).sort_values('importance', ascending=False)\n        \n        print(f\"\\nTop 10 features for {model_name}:\")\n        print(feature_importance.head(10))\n        \n        # Plot feature importance\n        plt.figure(figsize=(10, 8))\n        top_features = feature_importance.head(15)\n        plt.barh(range(len(top_features)), top_features['importance'])\n        plt.yticks(range(len(top_features)), top_features['feature'])\n        plt.xlabel('Feature Importance')\n        plt.title(f'Top 15 Feature Importance - {model_name}')\n        plt.gca().invert_yaxis()\n        plt.tight_layout()\n        plt.savefig(f'/home/ubuntu/plots/feature_importance_{model_name.replace(\" \", \"_\").lower()}.png', \n                   dpi=300, bbox_inches='tight')\n        plt.close()\n        \n        return feature_importance\n    else:\n        print(f\"Feature importance not available for {model_name}\")\n        return None\n\ndef create_prediction_plots(y_test, predictions, results):\n    \"\"\"Create prediction vs actual plots\"\"\"\n    \n    print(\"Creating prediction plots...\")\n    \n    # Create subplots for all models\n    n_models = len(predictions)\n    n_cols = 2\n    n_rows = (n_models + 1) // 2\n    \n    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))\n    if n_rows == 1:\n        axes = axes.reshape(1, -1)\n    \n    for i, (model_name, y_pred) in enumerate(predictions.items()):\n        row = i // n_cols\n        col = i % n_cols\n        \n        # Find corresponding R² score\n        r2 = next(r['test_r2'] for r in results if r['model_name'] == model_name)\n        rmse = next(r['test_rmse'] for r in results if r['model_name'] == model_name)\n        \n        axes[row, col].scatter(y_test, y_pred, alpha=0.5)\n        axes[row, col].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)\n        axes[row, col].set_xlabel('Actual Power (W)')\n        axes[row, col].set_ylabel('Predicted Power (W)')\n        axes[row, col].set_title(f'{model_name}\\nR² = {r2:.3f}, RMSE = {rmse:.1f}')\n        axes[row, col].grid(True, alpha=0.3)\n    \n    # Hide empty subplots\n    for i in range(n_models, n_rows * n_cols):\n        row = i // n_cols\n        col = i % n_cols\n        axes[row, col].set_visible(False)\n    \n    plt.tight_layout()\n    plt.savefig('/home/ubuntu/plots/model_predictions.png', dpi=300, bbox_inches='tight')\n    plt.close()\n\ndef create_model_comparison(results):\n    \"\"\"Create model comparison visualization\"\"\"\n    \n    print(\"Creating model comparison...\")\n    \n    # Convert results to DataFrame\n    df_results = pd.DataFrame(results)\n    \n    # Create comparison plots\n    fig, axes = plt.subplots(2, 2, figsize=(15, 12))\n    \n    # RMSE comparison\n    axes[0, 0].bar(df_results['model_name'], df_results['test_rmse'])\n    axes[0, 0].set_title('Test RMSE Comparison')\n    axes[0, 0].set_ylabel('RMSE')\n    axes[0, 0].tick_params(axis='x', rotation=45)\n    axes[0, 0].grid(True, alpha=0.3)\n    \n    # R² comparison\n    axes[0, 1].bar(df_results['model_name'], df_results['test_r2'])\n    axes[0, 1].set_title('Test R² Comparison')\n    axes[0, 1].set_ylabel('R²')\n    axes[0, 1].tick_params(axis='x', rotation=45)\n    axes[0, 1].grid(True, alpha=0.3)\n    \n    # MAE comparison\n    axes[1, 0].bar(df_results['model_name'], df_results['test_mae'])\n    axes[1, 0].set_title('Test MAE Comparison')\n    axes[1, 0].set_ylabel('MAE')\n    axes[1, 0].tick_params(axis='x', rotation=45)\n    axes[1, 0].grid(True, alpha=0.3)\n    \n    # Cross-validation RMSE\n    axes[1, 1].bar(df_results['model_name'], df_results['cv_rmse'])\n    axes[1, 1].set_title('Cross-Validation RMSE')\n    axes[1, 1].set_ylabel('CV RMSE')\n    axes[1, 1].tick_params(axis='x', rotation=45)\n    axes[1, 1].grid(True, alpha=0.3)\n    \n    plt.tight_layout()\n    plt.savefig('/home/ubuntu/plots/model_comparison.png', dpi=300, bbox_inches='tight')\n    plt.close()\n    \n    # Save results to CSV\n    df_results.to_csv('/home/ubuntu/model_results.csv', index=False)\n    \n    return df_results\n\ndef main():\n    \"\"\"Main function to run the complete model development pipeline\"\"\"\n    \n    print(\"SOLAR POWER PREDICTION MODEL DEVELOPMENT\")\n    print(\"=\"*50)\n    \n    # Create plots directory\n    import os\n    os.makedirs('/home/ubuntu/plots', exist_ok=True)\n    \n    # Load data\n    data, feature_info = load_processed_data()\n    \n    # Prepare data for modeling\n    X_train, X_test, y_train, y_test, scaler = prepare_model_data(data, feature_info)\n    \n    # Train baseline models\n    baseline_results, baseline_predictions = train_baseline_models(X_train, X_test, y_train, y_test)\n    \n    # Train advanced models\n    advanced_results, advanced_predictions = train_advanced_models(X_train, X_test, y_train, y_test)\n    \n    # Combine results\n    all_results = baseline_results + advanced_results\n    all_predictions = {**baseline_predictions, **advanced_predictions}\n    \n    # Hyperparameter tuning\n    best_rf, best_gb = hyperparameter_tuning(X_train, y_train)\n    \n    # Evaluate tuned models\n    tuned_rf_metrics, tuned_rf_pred = evaluate_model(best_rf, X_train, X_test, y_train, y_test, 'Random Forest (Tuned)')\n    tuned_gb_metrics, tuned_gb_pred = evaluate_model(best_gb, X_train, X_test, y_train, y_test, 'Gradient Boosting (Tuned)')\n    \n    all_results.extend([tuned_rf_metrics, tuned_gb_metrics])\n    all_predictions['Random Forest (Tuned)'] = tuned_rf_pred\n    all_predictions['Gradient Boosting (Tuned)'] = tuned_gb_pred\n    \n    # Feature importance analysis\n    analyze_feature_importance(best_rf, X_train.columns, 'Random Forest (Tuned)')\n    analyze_feature_importance(best_gb, X_train.columns, 'Gradient Boosting (Tuned)')\n    \n    # Create visualizations\n    create_prediction_plots(y_test, all_predictions, all_results)\n    df_results = create_model_comparison(all_results)\n    \n    # Save scaler\n    joblib.dump(scaler, '/home/ubuntu/scaler.pkl')\n    \n    print(\"\\n\" + \"=\"*50)\n    print(\"MODEL DEVELOPMENT COMPLETE\")\n    print(\"=\"*50)\n    print(\"\\nBest performing models:\")\n    best_models = df_results.nsmallest(3, 'test_rmse')\n    for _, row in best_models.iterrows():\n        print(f\"{row['model_name']}: RMSE = {row['test_rmse']:.2f}, R² = {row['test_r2']:.3f}\")\n    \n    print(\"\\nFiles saved:\")\n    print(\"- Model files: /home/ubuntu/model_*.pkl\")\n    print(\"- Scaler: /home/ubuntu/scaler.pkl\")\n    print(\"- Results: /home/ubuntu/model_results.csv\")\n    print(\"- Plots: /home/ubuntu/plots/\")\n    \n    return df_results\n\nif __name__ == \"__main__\":\n    results = main()

