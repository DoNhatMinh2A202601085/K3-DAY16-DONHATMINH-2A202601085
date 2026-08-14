#!/usr/bin/env python3
"""
Benchmark script for LightGBM on Credit Card Fraud Detection dataset.
Measures training time, inference latency/throughput, and classification metrics.
"""

import json
import time
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# Configuration
DATA_PATH = os.path.expanduser("~/ml-benchmark/creditcard.csv")
OUTPUT_PATH = os.path.expanduser("~/benchmark_result.json")
RANDOM_STATE = 42
TEST_SIZE = 0.2

def load_data():
    """Load the credit card fraud dataset."""
    print("Loading dataset...")
    start_time = time.time()
    df = pd.read_csv(DATA_PATH)
    load_time = time.time() - start_time
    print(f"Dataset loaded in {load_time:.4f} seconds")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:5]}... (total {len(df.columns)})")
    return df, load_time

def prepare_data(df):
    """Prepare features and target, split train/test."""
    print("\nPreparing data...")

    # Features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # Maintain class distribution
    )

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Fraud ratio in train: {y_train.mean():.4%}")
    print(f"Fraud ratio in test: {y_test.mean():.4%}")

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Train LightGBM classifier."""
    print("\nTraining LightGBM model...")

    # LightGBM parameters optimized for fraud detection
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': RANDOM_STATE,
        'n_jobs': -1,
        'is_unbalance': True  # Handle imbalanced classes
    }

    # Create dataset
    train_data = lgb.Dataset(X_train, label=y_train)

    # Train with early stopping
    start_time = time.time()
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[train_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.4f} seconds")
    print(f"Best iteration: {model.best_iteration}")

    return model, training_time

def evaluate_model(model, X_test, y_test):
    """Evaluate model and compute metrics."""
    print("\nEvaluating model...")

    # Get predictions
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics
    metrics = {
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
        'Accuracy': accuracy_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred)
    }

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics

def measure_inference(model, X_test):
    """Measure inference latency and throughput."""
    print("\nMeasuring inference performance...")

    # Use a single sample for latency measurement
    single_sample = X_test.iloc[[0]]

    # Warm-up run
    _ = model.predict(single_sample, num_iteration=model.best_iteration)

    # Measure latency (average of 100 runs)
    n_latency_runs = 100
    start_latency = time.time()
    for _ in range(n_latency_runs):
        _ = model.predict(single_sample, num_iteration=model.best_iteration)
    total_latency_time = time.time() - start_latency
    latency_per_sample = (total_latency_time / n_latency_runs) * 1000  # Convert to ms

    print(f"Inference latency (1 row): {latency_per_sample:.4f} ms (avg of {n_latency_runs} runs)")

    # Measure throughput (1000 rows)
    batch_size = 1000
    if len(X_test) >= batch_size:
        batch = X_test.iloc[:batch_size]
    else:
        batch = X_test

    # Warm-up run
    _ = model.predict(batch, num_iteration=model.best_iteration)

    # Measure throughput (average of 10 runs)
    n_throughput_runs = 10
    start_throughput = time.time()
    for _ in range(n_throughput_runs):
        _ = model.predict(batch, num_iteration=model.best_iteration)
    total_throughput_time = time.time() - start_throughput

    throughput_samples_per_sec = (batch.shape[0] * n_throughput_runs) / total_throughput_time
    throughput_ms_per_sample = (total_throughput_time / (batch.shape[0] * n_throughput_runs)) * 1000

    print(f"Inference throughput ({batch.shape[0]} rows): {throughput_samples_per_sec:.2f} samples/sec")
    print(f"({throughput_ms_per_sample:.4f} ms per sample)")

    return {
        'latency_ms': latency_per_sample,
        'throughput_samples_per_sec': throughput_samples_per_sec
    }

def save_results(load_time, training_time, best_iteration, metrics, inference_results):
    """Save all results to JSON file."""
    results = {
        'benchmark_info': {
            'dataset': 'Credit Card Fraud Detection',
            'model': 'LightGBM',
            'random_state': RANDOM_STATE,
            'test_size': TEST_SIZE
        },
        'timing': {
            'load_data_seconds': load_time,
            'training_seconds': training_time,
            'total_seconds': load_time + training_time
        },
        'model_info': {
            'best_iteration': best_iteration
        },
        'classification_metrics': {
            'AUC-ROC': metrics['AUC-ROC'],
            'Accuracy': metrics['Accuracy'],
            'F1-Score': metrics['F1-Score'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall']
        },
        'inference_performance': {
            'latency_ms_per_sample': inference_results['latency_ms'],
            'throughput_samples_per_sec': inference_results['throughput_samples_per_sec']
        }
    }

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {OUTPUT_PATH}")
    return results

def print_summary(results):
    """Print a formatted summary table."""
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print(f"{'Metric':<30} {'Value':>20}")
    print("-"*60)
    print(f"{'Thời gian load data (s):':<30} {results['timing']['load_data_seconds']:>20.4f}")
    print(f"{'Thời gian training (s):':<30} {results['timing']['training_seconds']:>20.4f}")
    print(f"{'Best iteration:':<30} {results['model_info']['best_iteration']:>20}")
    print("-"*60)
    print(f"{'AUC-ROC:':<30} {results['classification_metrics']['AUC-ROC']:>20.4f}")
    print(f"{'Accuracy:':<30} {results['classification_metrics']['Accuracy']:>20.4f}")
    print(f"{'F1-Score:':<30} {results['classification_metrics']['F1-Score']:>20.4f}")
    print(f"{'Precision:':<30} {results['classification_metrics']['Precision']:>20.4f}")
    print(f"{'Recall:':<30} {results['classification_metrics']['Recall']:>20.4f}")
    print("-"*60)
    print(f"{'Inference latency (1 row) ms:':<30} {results['inference_performance']['latency_ms_per_sample']:>20.4f}")
    print(f"{'Inference throughput (rows/s):':<30} {results['inference_performance']['throughput_samples_per_sec']:>20.2f}")
    print("="*60)

def main():
    """Main benchmark execution."""
    print("="*60)
    print("LightGBM Benchmark - Credit Card Fraud Detection")
    print("="*60)

    # Step 1: Load data
    df, load_time = load_data()

    # Step 2: Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Step 3: Train model
    model, training_time = train_model(X_train, y_train)

    # Step 4: Evaluate model
    metrics = evaluate_model(model, X_test, y_test)

    # Step 5: Measure inference performance
    inference_results = measure_inference(model, X_test)

    # Step 6: Save results
    results = save_results(
        load_time,
        training_time,
        model.best_iteration,
        metrics,
        inference_results
    )

    # Print summary
    print_summary(results)

    return results

if __name__ == "__main__":
    main()
