#!/usr/bin/env python3
"""
Run experiments with hyperparameter search for deep learning models
This script:
1. Runs hyperparameter search for each feature count experiment
2. Re-runs deep learning evaluation with optimized hyperparameters
3. Updates results with optimized model performance
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import shutil

# Import hyperparameter search
sys.path.insert(0, '/home/eo/allergen-prediction')
from comprehensive_hyperparameter_search import (
    run_hyperparameter_search_for_experiment,
    FlexibleMLP, FlexibleCNN, FlexibleLSTM, FlexibleTransformer,
    train_and_evaluate
)
from ml_evaluation.ml_evaluation import AllergenMLEvaluator
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_optimized_dl_evaluation(experiment_num, n_features, base_results_dir="/home/eo/allergen-prediction/results"):
    """Run deep learning evaluation with optimized hyperparameters"""
    print(f"\n{'='*80}")
    print(f"RUNNING OPTIMIZED DL EVALUATION FOR EXPERIMENT {experiment_num} ({n_features} features)")
    print(f"{'='*80}")
    
    exp_dir = f"{base_results_dir}/experiment{experiment_num}"
    hp_dir = f"{exp_dir}/hyperparameter_search"
    dl_dir = f"{exp_dir}/deep_learning_evaluation"
    os.makedirs(dl_dir, exist_ok=True)
    
    # Load optimized hyperparameters
    hp_file = f"{hp_dir}/best_hyperparameters.json"
    if not os.path.exists(hp_file):
        print(f"Warning: No hyperparameters found at {hp_file}. Skipping optimized evaluation.")
        return None
    
    with open(hp_file, 'r') as f:
        best_hps = json.load(f)
    
    # Load data
    pos_train = pd.read_csv(f"{exp_dir}/Pos_train_descriptors.csv")
    neg_train = pd.read_csv(f"{exp_dir}/Neg_train_descriptors.csv")
    pos_test = pd.read_csv(f"{exp_dir}/Pos_test_descriptors.csv")
    neg_test = pd.read_csv(f"{exp_dir}/Neg_test_descriptors.csv")
    
    train_data = pd.concat([pos_train, neg_train], ignore_index=True)
    test_data = pd.concat([pos_test, neg_test], ignore_index=True)
    
    X_train_full = train_data.drop(['Name'], axis=1).values
    y_train = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])
    X_test_full = test_data.drop(['Name'], axis=1).values
    y_test = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))])
    
    # Handle missing values
    X_train_full = np.nan_to_num(X_train_full, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_full = np.nan_to_num(X_test_full, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test_full)
    
    # Split for validation
    X_train, X_val, y_train_split, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    input_dim = X_train.shape[1]
    results = []
    
    # Train each model with optimized hyperparameters
    model_mapping = {
        'MLP': 'Basic MLP',
        'CNN': 'CNN',
        'LSTM': 'LSTM',
        'Transformer': 'Transformer'
    }
    
    for model_key, model_display_name in model_mapping.items():
        if model_key not in best_hps:
            print(f"Skipping {model_display_name} - no hyperparameters found")
            continue
        
        print(f"\nTraining {model_display_name} with optimized hyperparameters...")
        hp_params = best_hps[model_key]['params']
        hp_metrics = best_hps[model_key]['metrics']
        
        try:
            # Create model with optimized architecture
            if model_key == 'MLP':
                model = FlexibleMLP(
                    input_dim,
                    hp_params['hidden_sizes'],
                    hp_params['dropout_rate'],
                    hp_params['activation']
                ).to(device)
            elif model_key == 'CNN':
                model = FlexibleCNN(
                    input_dim,
                    hp_params['num_filters'],
                    hp_params['kernel_sizes'],
                    hp_params['fc_sizes'],
                    hp_params['dropout_rate']
                ).to(device)
            elif model_key == 'LSTM':
                model = FlexibleLSTM(
                    input_dim,
                    hp_params['hidden_sizes'],
                    hp_params['num_layers'],
                    hp_params['dropout_rate'],
                    hp_params['use_attention']
                ).to(device)
            elif model_key == 'Transformer':
                model = FlexibleTransformer(
                    input_dim,
                    hp_params['d_model'],
                    hp_params['nhead'],
                    hp_params['num_layers'],
                    hp_params['dim_feedforward'],
                    hp_params['dropout_rate']
                ).to(device)
            
            # Create data loaders
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train_split))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))
            
            batch_size = hp_params.get('batch_size', 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Train with optimized hyperparameters
            learning_rate = hp_params.get('learning_rate', 0.001)
            weight_decay = hp_params.get('weight_decay', 0.0)
            
            metrics, _ = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                learning_rate, weight_decay, epochs=100
            )
            
            results.append({
                'Model': model_display_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC': metrics['auc']
            })
            
            print(f"  {model_display_name}: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}")
            
        except Exception as e:
            print(f"  Error training {model_display_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('AUC', ascending=False)
        results_df.to_csv(f"{dl_dir}/performance_summary_optimized.csv", index=False)
        
        # Also update the main performance summary if it exists
        main_summary_file = f"{dl_dir}/performance_summary.csv"
        if os.path.exists(main_summary_file):
            # Backup original
            shutil.copy(main_summary_file, f"{main_summary_file}.backup")
        
        results_df.to_csv(main_summary_file, index=False)
        print(f"\nOptimized results saved to {main_summary_file}")
    
    return results

def create_feature_selected_descriptors(experiment_num, n_features, base_results_dir="/home/eo/allergen-prediction/results"):
    """Create feature-selected descriptor files for an experiment if they don't exist"""
    exp_dir = f"{base_results_dir}/experiment{experiment_num}"
    descriptor_file = f"{exp_dir}/Pos_train_descriptors.csv"
    
    # Check if files already exist
    if os.path.exists(descriptor_file):
        print(f"Experiment {experiment_num}: Descriptor files already exist, skipping creation...")
        return True
    
    print(f"Creating feature-selected descriptor files for experiment {experiment_num} ({n_features} features)...")
    
    # Check if base descriptor files exist
    base_pos_train = f"{base_results_dir}/Pos_train_descriptors.csv"
    base_neg_train = f"{base_results_dir}/Neg_train_descriptors.csv"
    base_pos_test = f"{base_results_dir}/Pos_test_descriptors.csv"
    base_neg_test = f"{base_results_dir}/Neg_test_descriptors.csv"
    
    if not all([os.path.exists(f) for f in [base_pos_train, base_neg_train, base_pos_test, base_neg_test]]):
        print(f"Error: Base descriptor files not found in {base_results_dir}")
        print("Required files:")
        print(f"  - {base_pos_train}")
        print(f"  - {base_neg_train}")
        print(f"  - {base_pos_test}")
        print(f"  - {base_neg_test}")
        print("\nPlease run convert_sdf_to_descriptors.py first to create base descriptor files.")
        return False
    
    try:
        # Import feature selection function
        sys.path.insert(0, '/home/eo/allergen-prediction')
        from run_feature_count_experiments import select_features
        
        # Load base data
        pos_train = pd.read_csv(f"{base_results_dir}/Pos_train_descriptors.csv")
        neg_train = pd.read_csv(f"{base_results_dir}/Neg_train_descriptors.csv")
        pos_test = pd.read_csv(f"{base_results_dir}/Pos_test_descriptors.csv")
        neg_test = pd.read_csv(f"{base_results_dir}/Neg_test_descriptors.csv")
        
        # Combine and prepare
        train_data = pd.concat([pos_train, neg_train], ignore_index=True)
        test_data = pd.concat([pos_test, neg_test], ignore_index=True)
        
        X_train_full = train_data.drop(['Name'], axis=1)
        y_train = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])
        X_test_full = test_data.drop(['Name'], axis=1)
        y_test = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))])
        
        # Handle missing values
        X_train_full = X_train_full.replace([np.inf, -np.inf], np.nan)
        X_test_full = X_test_full.replace([np.inf, -np.inf], np.nan)
        X_train_full = X_train_full.fillna(X_train_full.median())
        X_test_full = X_test_full.fillna(X_train_full.median())
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_full)
        X_test_scaled = scaler.transform(X_test_full)
        
        # Select features
        feature_indices, selected_feature_names = select_features(
            X_train_scaled, y_train,
            list(X_train_full.columns),
            n_features,
            method='rf_importance'
        )
        
        # Create experiment directory
        os.makedirs(exp_dir, exist_ok=True)
        
        # For training data, reconstruct labels
        train_labels = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])
        train_pos_mask = train_labels == 1
        train_neg_mask = train_labels == 0
        
        test_labels = np.concatenate([np.ones(len(pos_test)), np.zeros(len(neg_test))])
        test_pos_mask = test_labels == 1
        test_neg_mask = test_labels == 0
        
        # Get names
        train_names = pd.concat([pos_train['Name'], neg_train['Name']], ignore_index=True).values
        test_names = pd.concat([pos_test['Name'], neg_test['Name']], ignore_index=True).values
        
        pos_train_selected = pd.DataFrame(
            X_train_scaled[train_pos_mask][:, feature_indices],
            columns=selected_feature_names
        )
        pos_train_selected.insert(0, 'Name', train_names[train_pos_mask])
        
        neg_train_selected = pd.DataFrame(
            X_train_scaled[train_neg_mask][:, feature_indices],
            columns=selected_feature_names
        )
        neg_train_selected.insert(0, 'Name', train_names[train_neg_mask])
        
        pos_test_selected = pd.DataFrame(
            X_test_scaled[test_pos_mask][:, feature_indices],
            columns=selected_feature_names
        )
        pos_test_selected.insert(0, 'Name', test_names[test_pos_mask])
        
        neg_test_selected = pd.DataFrame(
            X_test_scaled[test_neg_mask][:, feature_indices],
            columns=selected_feature_names
        )
        neg_test_selected.insert(0, 'Name', test_names[test_neg_mask])
        
        # Save to experiment directory
        pos_train_selected.to_csv(f"{exp_dir}/Pos_train_descriptors.csv", index=False)
        neg_train_selected.to_csv(f"{exp_dir}/Neg_train_descriptors.csv", index=False)
        pos_test_selected.to_csv(f"{exp_dir}/Pos_test_descriptors.csv", index=False)
        neg_test_selected.to_csv(f"{exp_dir}/Neg_test_descriptors.csv", index=False)
        
        # Save selected features info
        feature_selection_df = pd.DataFrame({
            'Feature_Name': selected_feature_names,
            'Original_Index': feature_indices,
            'Rank': range(1, len(feature_indices) + 1)
        })
        feature_selection_df.to_csv(f"{exp_dir}/selected_features.csv", index=False)
        
        print(f"Successfully created descriptor files for experiment {experiment_num}")
        return True
        
    except Exception as e:
        print(f"Error creating descriptor files for experiment {experiment_num}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution"""
    print("="*80)
    print("EXPERIMENTS WITH HYPERPARAMETER SEARCH")
    print("="*80)
    
    base_results_dir = "/home/eo/allergen-prediction/results"
    feature_counts = [50, 100, 150, 200, 217]
    
    # Step 0: Ensure feature-selected descriptor files exist
    print("\nSTEP 0: Checking and creating feature-selected descriptor files...")
    for exp_num, n_features in enumerate(feature_counts, start=1):
        create_feature_selected_descriptors(exp_num, n_features, base_results_dir)
    
    # Step 1: Run hyperparameter search for each experiment
    print("\nSTEP 1: Running hyperparameter search...")
    for exp_num, n_features in enumerate(feature_counts, start=1):
        exp_dir = f"{base_results_dir}/experiment{exp_num}"
        hp_dir = f"{exp_dir}/hyperparameter_search"
        
        # Check if hyperparameter search already done
        if os.path.exists(f"{hp_dir}/best_hyperparameters.json"):
            print(f"Experiment {exp_num}: Hyperparameter search already completed, skipping...")
            continue
        
        print(f"\nRunning hyperparameter search for experiment {exp_num} ({n_features} features)...")
        try:
            run_hyperparameter_search_for_experiment(exp_num, n_features, base_results_dir)
        except Exception as e:
            print(f"Hyperparameter search failed for experiment {exp_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 2: Run optimized DL evaluation
    print("\n\nSTEP 2: Running optimized deep learning evaluation...")
    all_results = {}
    for exp_num, n_features in enumerate(feature_counts, start=1):
        try:
            results = run_optimized_dl_evaluation(exp_num, n_features, base_results_dir)
            if results:
                all_results[exp_num] = results
        except Exception as e:
            print(f"Optimized evaluation failed for experiment {exp_num}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Step 3: Create summary
    print("\n\nSTEP 3: Creating summary...")
    summary_data = []
    for exp_num, results in all_results.items():
        n_features = feature_counts[exp_num - 1]
        for result in results:
            summary_data.append({
                'Experiment': exp_num,
                'N_Features': n_features,
                'Model': result['Model'],
                'AUC': result['AUC'],
                'Accuracy': result['Accuracy'],
                'Precision': result['Precision'],
                'Recall': result['Recall'],
                'F1-Score': result['F1-Score']
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{base_results_dir}/optimized_dl_results_summary.csv", index=False)
        
        print("\n" + "="*80)
        print("OPTIMIZED DEEP LEARNING RESULTS SUMMARY")
        print("="*80)
        print(summary_df.groupby(['N_Features', 'Model'])[['AUC', 'Accuracy']].max().unstack())
        print("="*80)
    
    print("\nAll experiments with hyperparameter optimization complete!")

if __name__ == "__main__":
    import shutil
    main()

