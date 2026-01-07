#!/usr/bin/env python3
"""
Phase 2: Comparative Evaluation with Optimized Architectures
Re-runs evaluation using best hyperparameters from Phase 1, with bug fixes
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Import model classes and training function
from comprehensive_hyperparameter_search import (
    FlexibleMLP, FlexibleCNN, FlexibleLSTM, FlexibleTransformer,
    train_and_evaluate, device
)

def calculate_cnn_output_size(input_dim, num_filters, kernel_sizes):
    """Dynamically calculate CNN output size by running a dummy forward pass"""
    # Create a dummy model to calculate output size
    dummy_model = FlexibleCNN(input_dim, num_filters, kernel_sizes, [64], 0.1)
    dummy_input = torch.randn(1, input_dim)
    with torch.no_grad():
        dummy_output = dummy_model.convs[0](dummy_input.unsqueeze(1))
        for i in range(1, len(dummy_model.convs)):
            dummy_output = dummy_model.convs[i](dummy_output)
            dummy_output = dummy_model.pools[i](dummy_output)
        output_size = dummy_output.view(1, -1).size(1)
    return output_size

class FixedFlexibleCNN(nn.Module):
    """Fixed CNN with dynamic output size calculation"""
    def __init__(self, input_dim, num_filters, kernel_sizes, fc_sizes, dropout_rate):
        super(FixedFlexibleCNN, self).__init__()
        self.input_dim = input_dim
        
        # Convolutional layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_channels = 1
        for i, (out_channels, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size, 
                                      padding=kernel_size//2))
            self.bns.append(nn.BatchNorm1d(out_channels))
            self.pools.append(nn.MaxPool1d(2))
            in_channels = out_channels
        
        # Dynamically calculate flattened size by running a dummy forward pass
        # We need to do this after creating conv layers but before FC layers
        dummy_input = torch.randn(1, 1, input_dim)
        with torch.no_grad():
            x = dummy_input
            for conv, bn, pool in zip(self.convs, self.bns, self.pools):
                x = torch.relu(bn(conv(x)))
                x = pool(x)
            conv_output_size = x.view(1, -1).size(1)
        
        # Ensure minimum size
        if conv_output_size <= 0:
            raise ValueError(f"Invalid CNN output size: {conv_output_size} for input_dim={input_dim}")
        
        # Fully connected layers
        self.fcs = nn.ModuleList()
        prev_size = conv_output_size
        for fc_size in fc_sizes:
            self.fcs.append(nn.Linear(prev_size, fc_size))
            prev_size = fc_size
        self.fcs.append(nn.Linear(prev_size, 1))
        
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        x = x.unsqueeze(1)
        for conv, bn, pool in zip(self.convs, self.bns, self.pools):
            x = torch.relu(bn(conv(x)))
            x = pool(x)
        x = x.view(x.size(0), -1)
        for i, fc in enumerate(self.fcs[:-1]):
            x = torch.relu(fc(x))
            x = self.dropout(x)
        x = torch.sigmoid(self.fcs[-1](x))
        return x

def run_phase2_evaluation(experiment_num, n_features, base_results_dir="/home/eo/allergen-prediction/results"):
    """Run Phase 2 comparative evaluation with optimized hyperparameters"""
    print(f"\n{'='*80}")
    print(f"PHASE 2: COMPARATIVE EVALUATION - EXPERIMENT {experiment_num} ({n_features} features)")
    print(f"{'='*80}")
    
    exp_dir = f"{base_results_dir}/experiment{experiment_num}"
    hp_dir = f"{exp_dir}/hyperparameter_search"
    phase2_dir = f"{base_results_dir}/phase2_comparative_evaluation"
    os.makedirs(phase2_dir, exist_ok=True)
    exp_phase2_dir = f"{phase2_dir}/experiment{experiment_num}"
    os.makedirs(exp_phase2_dir, exist_ok=True)
    
    # Load optimized hyperparameters from Phase 1
    hp_file = f"{hp_dir}/best_hyperparameters.json"
    if not os.path.exists(hp_file):
        print(f"Warning: No hyperparameters found at {hp_file}. Skipping.")
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
    print(f"Input dimension: {input_dim}")
    print(f"Training samples: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test_scaled)}")
    
    results = []
    model_mapping = {
        'MLP': 'MLP',
        'CNN': 'CNN',
        'LSTM': 'LSTM',
        'Transformer': 'Transformer'
    }
    
    # Train each model with optimized hyperparameters
    for model_key, model_display_name in model_mapping.items():
        if model_key not in best_hps:
            print(f"Skipping {model_display_name} - no hyperparameters found")
            continue
        
        print(f"\nTraining {model_display_name} with optimized hyperparameters...")
        hp_params = best_hps[model_key]['params']
        hp_metrics = best_hps[model_key]['metrics']
        
        print(f"  Best HP search AUC: {hp_metrics['auc']:.4f}")
        
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
                # Use fixed CNN with dynamic size calculation
                model = FixedFlexibleCNN(
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
                    hp_params.get('use_attention', False)
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
            
            # Create data loaders - FIX: Use FloatTensor for labels (BCELoss expects float)
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train_split))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
            test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))
            
            batch_size = hp_params.get('batch_size', 32)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Train with optimized hyperparameters
            learning_rate = hp_params.get('learning_rate', 0.001)
            weight_decay = hp_params.get('weight_decay', 0.0)
            
            metrics, model_state = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                learning_rate, weight_decay, epochs=100
            )
            
            # Verify metrics are reasonable
            if metrics['auc'] < 0.5 or metrics['precision'] == 0.0:
                print(f"  WARNING: {model_display_name} produced suspicious results!")
                print(f"    AUC: {metrics['auc']:.4f}, Precision: {metrics['precision']:.4f}")
            
            results.append({
                'Model': model_display_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1'],
                'AUC': metrics['auc']
            })
            
            print(f"  {model_display_name}: AUC={metrics['auc']:.4f}, Acc={metrics['accuracy']:.4f}, "
                  f"Prec={metrics['precision']:.4f}, Rec={metrics['recall']:.4f}")
            
            # Save model state
            if model_state:
                torch.save(model_state, f"{exp_phase2_dir}/{model_key.lower()}_model.pt")
            
        except Exception as e:
            print(f"  ERROR training {model_display_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('AUC', ascending=False)
        results_df.to_csv(f"{exp_phase2_dir}/performance_summary.csv", index=False)
        
        print(f"\nResults saved to: {exp_phase2_dir}/performance_summary.csv")
        print("\nPerformance Summary:")
        print(results_df.to_string(index=False))
    
    return results

def main():
    """Run Phase 2 evaluation for all experiments"""
    print("="*80)
    print("PHASE 2: COMPARATIVE EVALUATION WITH OPTIMIZED ARCHITECTURES")
    print("="*80)
    
    feature_counts = [50, 100, 150, 200, 217]
    base_results_dir = "/home/eo/allergen-prediction/results"
    
    all_results = []
    
    for exp_num, n_features in enumerate(feature_counts, start=1):
        try:
            results = run_phase2_evaluation(exp_num, n_features, base_results_dir)
            if results:
                for r in results:
                    r['Experiment'] = exp_num
                    r['N_Features'] = n_features
                    all_results.append(r)
        except Exception as e:
            print(f"Experiment {exp_num} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall summary
    if all_results:
        overall_df = pd.DataFrame(all_results)
        overall_df = overall_df.sort_values(['N_Features', 'AUC'], ascending=[True, False])
        summary_file = f"{base_results_dir}/phase2_comparative_evaluation/overall_summary.csv"
        overall_df.to_csv(summary_file, index=False)
        
        print("\n" + "="*80)
        print("PHASE 2 EVALUATION COMPLETE")
        print("="*80)
        print(f"\nOverall summary saved to: {summary_file}")
        print("\nBest model per feature count:")
        for n_feat in feature_counts:
            feat_data = overall_df[overall_df['N_Features'] == n_feat]
            if len(feat_data) > 0:
                best = feat_data.loc[feat_data['AUC'].idxmax()]
                print(f"  {n_feat} features: {best['Model']} (AUC={best['AUC']:.4f}, Acc={best['Accuracy']:.4f})")

if __name__ == "__main__":
    main()

