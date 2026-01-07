#!/usr/bin/env python3
"""
Run experiments with different feature counts
Each experiment is saved in a separate subdirectory
"""

import os
import sys
import shutil
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# Import the evaluator classes
sys.path.insert(0, '/home/eo/allergen-prediction')
from ml_evaluation.ml_evaluation import AllergenMLEvaluator
from deep_learning_evaluation.evaluation import AllergenPyTorchEvaluator

def select_features(X_train, y_train, feature_names, n_features, method='rf_importance'):
    """
    Select top N features using specified method
    
    Args:
        X_train: Training features
        y_train: Training labels
        feature_names: List of feature names
        n_features: Number of features to select
        method: 'rf_importance' or 'mutual_info'
    
    Returns:
        selected_indices: Indices of selected features
        selected_names: Names of selected features
    """
    if method == 'rf_importance':
        # Use Random Forest feature importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
    elif method == 'mutual_info':
        # Use mutual information
        mi_scores = mutual_info_classif(X_train, y_train, random_state=42, n_jobs=-1)
        indices = np.argsort(mi_scores)[::-1][:n_features]
    else:
        raise ValueError(f"Unknown method: {method}")
    
    selected_names = [feature_names[i] for i in indices]
    return indices, selected_names

def run_experiment_with_features(n_features, experiment_num, base_results_dir="/home/eo/allergen-prediction/results"):
    """
    Run ML and DL evaluations with a specific number of features
    
    Args:
        n_features: Number of features to use
        experiment_num: Unique experiment number
        base_results_dir: Base results directory
    """
    print("\n" + "="*80)
    print(f"EXPERIMENT {experiment_num}: Using {n_features} features")
    print("="*80)
    
    # Create experiment directory
    exp_dir = f"{base_results_dir}/experiment{experiment_num}"
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/ml_evaluation", exist_ok=True)
    os.makedirs(f"{exp_dir}/deep_learning_evaluation", exist_ok=True)
    
    # Load data
    print("Loading data...")
    pos_train = pd.read_csv(f"{base_results_dir}/Pos_train_descriptors.csv")
    neg_train = pd.read_csv(f"{base_results_dir}/Neg_train_descriptors.csv")
    pos_test = pd.read_csv(f"{base_results_dir}/Pos_test_descriptors.csv")
    neg_test = pd.read_csv(f"{base_results_dir}/Neg_test_descriptors.csv")
    
    # Add labels
    pos_train['label'] = 1
    neg_train['label'] = 0
    pos_test['label'] = 1
    neg_test['label'] = 0
    
    # Combine datasets
    train_data = pd.concat([pos_train, neg_train], ignore_index=True)
    test_data = pd.concat([pos_test, neg_test], ignore_index=True)
    
    # Separate features from labels
    X_train_full = train_data.drop(['Name', 'label'], axis=1)
    y_train = train_data['label']
    X_test_full = test_data.drop(['Name', 'label'], axis=1)
    y_test = test_data['label']
    
    # Handle missing values
    X_train_full = X_train_full.replace([np.inf, -np.inf], np.nan)
    X_test_full = X_test_full.replace([np.inf, -np.inf], np.nan)
    X_train_full = X_train_full.fillna(X_train_full.median())
    X_test_full = X_test_full.fillna(X_train_full.median())
    
    # Scale features (before selection)
    scaler = StandardScaler()
    X_train_scaled_full = scaler.fit_transform(X_train_full)
    X_test_scaled_full = scaler.transform(X_test_full)
    
    # Select features
    print(f"Selecting top {n_features} features using Random Forest importance...")
    feature_indices, selected_feature_names = select_features(
        X_train_scaled_full, y_train, 
        list(X_train_full.columns), 
        n_features, 
        method='rf_importance'
    )
    
    # Apply feature selection
    X_train_selected = X_train_scaled_full[:, feature_indices]
    X_test_selected = X_test_scaled_full[:, feature_indices]
    
    print(f"Selected features shape: {X_train_selected.shape}")
    
    # Save feature selection info
    feature_selection_df = pd.DataFrame({
        'Feature_Name': selected_feature_names,
        'Original_Index': feature_indices,
        'Rank': range(1, len(feature_indices) + 1)
    })
    feature_selection_df.to_csv(f"{exp_dir}/selected_features.csv", index=False)
    
    # Create temporary descriptor CSV files with selected features
    # Split back into pos/neg for the evaluator classes
    train_pos_mask = train_data['label'] == 1
    train_neg_mask = train_data['label'] == 0
    test_pos_mask = test_data['label'] == 1
    test_neg_mask = test_data['label'] == 0
    
    # Create DataFrames with selected features
    pos_train_selected = pd.DataFrame(
        X_train_selected[train_pos_mask], 
        columns=selected_feature_names
    )
    pos_train_selected.insert(0, 'Name', train_data.loc[train_pos_mask, 'Name'].values)
    
    neg_train_selected = pd.DataFrame(
        X_train_selected[train_neg_mask],
        columns=selected_feature_names
    )
    neg_train_selected.insert(0, 'Name', train_data.loc[train_neg_mask, 'Name'].values)
    
    pos_test_selected = pd.DataFrame(
        X_test_selected[test_pos_mask],
        columns=selected_feature_names
    )
    pos_test_selected.insert(0, 'Name', test_data.loc[test_pos_mask, 'Name'].values)
    
    neg_test_selected = pd.DataFrame(
        X_test_selected[test_neg_mask],
        columns=selected_feature_names
    )
    neg_test_selected.insert(0, 'Name', test_data.loc[test_neg_mask, 'Name'].values)
    
    # Save to CSV files
    pos_train_selected.to_csv(f"{exp_dir}/Pos_train_descriptors.csv", index=False)
    neg_train_selected.to_csv(f"{exp_dir}/Neg_train_descriptors.csv", index=False)
    pos_test_selected.to_csv(f"{exp_dir}/Pos_test_descriptors.csv", index=False)
    neg_test_selected.to_csv(f"{exp_dir}/Neg_test_descriptors.csv", index=False)
    
    # Run ML Evaluation
    print("\n" + "-"*80)
    print("Running ML Evaluation...")
    print("-"*80)
    
    try:
        ml_evaluator = AllergenMLEvaluator(results_dir=exp_dir)
        ml_evaluator.eval_dir = f"{exp_dir}/ml_evaluation"
        
        # Initialize and train models
        X_train, X_test, y_train_eval, y_test_eval, feature_names_eval = ml_evaluator.load_data()
        ml_evaluator.initialize_models()
        ml_evaluator.train_models(X_train, y_train_eval)
        
        # Evaluate models
        ml_evaluator.evaluate_models(X_test, y_test_eval)
        
        # Create plots
        ml_evaluator.create_performance_plots(y_test_eval)
        ml_evaluator.create_feature_importance_plot(feature_names_eval)
        
        # Generate summary
        ml_results = ml_evaluator.generate_summary_report()
        print(f"ML Evaluation complete for {n_features} features")
    except Exception as e:
        print(f"ML Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        ml_results = None
    
    # Run Deep Learning Evaluation
    print("\n" + "-"*80)
    print("Running Deep Learning Evaluation...")
    print("-"*80)
    
    try:
        dl_evaluator = AllergenPyTorchEvaluator(results_dir=exp_dir)
        dl_evaluator.eval_dir = f"{exp_dir}/deep_learning_evaluation"
        
        # Load data (will use the CSV files we created)
        X_train, X_test, y_train_eval, y_test_eval, feature_names_eval = dl_evaluator.load_data()
        
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train_eval, test_size=0.2, random_state=42, stratify=y_train_eval
        )
        
        # Initialize and train models
        dl_evaluator.initialize_models(X_train.shape[1])
        dl_evaluator.train_models(X_train_split, y_train_split, X_val, y_val)
        
        # Evaluate models
        dl_evaluator.evaluate_models(X_test, y_test_eval)
        
        # Create plots
        dl_evaluator.create_training_plots()
        dl_evaluator.create_performance_plots(y_test_eval)
        
        # Generate summary
        dl_results = dl_evaluator.generate_summary_report()
        print(f"Deep Learning Evaluation complete for {n_features} features")
    except Exception as e:
        print(f"Deep Learning Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        dl_results = None
    
    # Save experiment summary
    summary = {
        'experiment_num': experiment_num,
        'n_features': n_features,
        'ml_best_auc': ml_results.iloc[0]['AUC'] if ml_results is not None and len(ml_results) > 0 else None,
        'ml_best_model': ml_results.iloc[0]['Model'] if ml_results is not None and len(ml_results) > 0 else None,
        'dl_best_auc': dl_results.iloc[0]['AUC'] if dl_results is not None and len(dl_results) > 0 else None,
        'dl_best_model': dl_results.iloc[0]['Model'] if dl_results is not None and len(dl_results) > 0 else None,
    }
    
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(f"{exp_dir}/experiment_summary.csv", index=False)
    
    print(f"\nExperiment {experiment_num} complete! Results saved to {exp_dir}")
    
    return summary

def main():
    """Run experiments with different feature counts"""
    print("="*80)
    print("FEATURE COUNT EXPERIMENTS")
    print("="*80)
    
    # Define feature counts to test
    feature_counts = [50, 100, 150, 200, 217]  # 217 is all features
    
    base_results_dir = "/home/eo/allergen-prediction/results"
    
    all_summaries = []
    
    for exp_num, n_features in enumerate(feature_counts, start=1):
        try:
            summary = run_experiment_with_features(n_features, exp_num, base_results_dir)
            all_summaries.append(summary)
        except Exception as e:
            print(f"Experiment {exp_num} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Create overall comparison
    if all_summaries:
        comparison_df = pd.DataFrame(all_summaries)
        comparison_df.to_csv(f"{base_results_dir}/feature_count_comparison.csv", index=False)
        
        print("\n" + "="*80)
        print("FEATURE COUNT COMPARISON SUMMARY")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)
    
    print("\nAll experiments complete!")

if __name__ == "__main__":
    main()

