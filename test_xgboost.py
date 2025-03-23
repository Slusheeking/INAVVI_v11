#!/usr/bin/env python3
"""
Enhanced XGBoost GPU Test Script

This script tests XGBoost's GPU acceleration capabilities by:
1. Comparing CPU vs GPU training performance
2. Evaluating model accuracy
3. Testing different tree methods
4. Measuring inference speed
"""

import xgboost as xgb
import numpy as np
import time
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set up logging
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('xgboost_test')


def print_separator(title):
    """Print a section separator with title"""
    width = 80
    print("\n" + "=" * width)
    print(f"{title.center(width)}")
    print("=" * width)


def test_xgboost_version():
    """Test and print XGBoost version information"""
    print_separator("XGBoost Version Information")
    print(f"XGBoost version: {xgb.__version__}")

    # Check if GPU is available
    try:
        # Create a small test DMatrix
        X = np.random.rand(10, 5)
        y = np.random.randint(0, 2, 10)
        dtrain = xgb.DMatrix(X, label=y)

        # Try to train with GPU
        param = {"tree_method": "gpu_hist", "gpu_id": 0}
        bst = xgb.train(param, dtrain, num_boost_round=1)
        print("GPU support: Available")
        print("CUDA available: Yes")
    except Exception as e:
        print(f"GPU support: Not available")
        print(f"CUDA available: No")
        print(f"Error: {e}")


def generate_dataset(n_samples=50000, n_features=20):
    """Generate a synthetic dataset for testing"""
    print_separator("Generating Dataset")
    print(f"Samples: {n_samples}, Features: {n_features}")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=2,
        random_state=42
    )

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(
        f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test


def benchmark_training(X_train, y_train, X_test, y_test):
    """Benchmark XGBoost training on CPU vs GPU"""
    print_separator("Training Performance Benchmark: CPU vs GPU")

    # Convert to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Common parameters
    common_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1
    }

    # Define tree methods to test
    tree_methods = [
        ('hist', 'CPU - Histogram'),
        ('gpu_hist', 'GPU - Histogram')
    ]

    results = {}

    # Test each tree method
    for method, name in tree_methods:
        try:
            params = common_params.copy()
            params['tree_method'] = method

            if 'gpu' in method:
                params['gpu_id'] = 0

            print(f"\nTesting: {name}")

            # Training
            start_time = time.time()
            model = xgb.train(
                params,
                dtrain,
                num_boost_round=100,
                evals=[(dtrain, 'train'), (dtest, 'test')],
                verbose_eval=25
            )
            train_time = time.time() - start_time

            # Prediction
            start_time = time.time()
            y_pred = model.predict(dtest)
            predict_time = time.time() - start_time

            # Convert probabilities to binary predictions
            y_pred_binary = np.round(y_pred)

            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred_binary)
            precision = precision_score(y_test, y_pred_binary)
            recall = recall_score(y_test, y_pred_binary)
            f1 = f1_score(y_test, y_pred_binary)

            results[name] = {
                'training_time': train_time,
                'prediction_time': predict_time,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

            print(f"Training time: {train_time:.4f} seconds")
            print(f"Prediction time: {predict_time:.4f} seconds")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")

        except Exception as e:
            print(f"Error testing {name}: {e}")

    # Compare results if we have both CPU and GPU
    if len(results) > 1:
        print("\nPerformance Comparison:")
        if 'CPU - Histogram' in results and 'GPU - Histogram' in results:
            cpu_time = results['CPU - Histogram']['training_time']
            gpu_time = results['GPU - Histogram']['training_time']
            speedup = cpu_time / gpu_time
            print(f"GPU Training Speedup: {speedup:.2f}x faster than CPU")

            cpu_pred_time = results['CPU - Histogram']['prediction_time']
            gpu_pred_time = results['GPU - Histogram']['prediction_time']
            pred_speedup = cpu_pred_time / gpu_pred_time
            print(
                f"GPU Prediction Speedup: {pred_speedup:.2f}x faster than CPU")

    return results


def test_feature_importance(X_train, y_train):
    """Test feature importance visualization"""
    print_separator("Feature Importance Analysis")

    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = {
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    }

    try:
        model = xgb.train(params, dtrain, num_boost_round=10)

        # Get feature importance
        importance = model.get_score(importance_type='weight')

        # Sort features by importance
        sorted_importance = sorted(
            importance.items(), key=lambda x: x[1], reverse=True)

        # Print top 10 features
        print("Top 10 important features:")
        for i, (feature, score) in enumerate(sorted_importance[:10], 1):
            print(f"{i}. Feature {feature}: {score}")

    except Exception as e:
        print(f"Error in feature importance analysis: {e}")


def main():
    """Main test function"""
    print_separator("XGBoost GPU Test")

    # Test XGBoost version
    test_xgboost_version()

    # Generate dataset
    X_train, X_test, y_train, y_test = generate_dataset()

    # Benchmark training
    results = benchmark_training(X_train, y_train, X_test, y_test)

    # Test feature importance
    test_feature_importance(X_train, y_train)

    print_separator("Test Complete")

    # Final summary
    if 'GPU - Histogram' in results:
        print("XGBoost GPU acceleration is working correctly!")
    else:
        print("XGBoost GPU acceleration test failed or not available.")


if __name__ == "__main__":
    main()
