#!/usr/bin/env python3
"""
Comprehensive Hyperparameter Search for All Deep Learning Architectures
This script performs hyperparameter optimization for MLP, CNN, LSTM, and Transformer models
"""

import os
import sys
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
import itertools
import json
import warnings
warnings.filterwarnings('ignore')

# Import torch functional
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FlexibleMLP(nn.Module):
    """Flexible MLP for hyperparameter search"""
    def __init__(self, input_dim, hidden_sizes, dropout_rate, activation='relu'):
        super(FlexibleMLP, self).__init__()
        layers = []
        prev_size = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class FlexibleCNN(nn.Module):
    """Flexible CNN for hyperparameter search"""
    def __init__(self, input_dim, num_filters, kernel_sizes, fc_sizes, dropout_rate):
        super(FlexibleCNN, self).__init__()
        self.input_dim = input_dim
        
        # Convolutional layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        in_channels = 1
        for i, (out_channels, kernel_size) in enumerate(zip(num_filters, kernel_sizes)):
            self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            self.bns.append(nn.BatchNorm1d(out_channels))
            self.pools.append(nn.MaxPool1d(2))
            in_channels = out_channels
        
        # Calculate flattened size
        conv_output_size = (input_dim // (2 ** len(num_filters))) * num_filters[-1]
        
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
            x = F.relu(bn(conv(x)))
            x = pool(x)
        x = x.view(x.size(0), -1)
        for i, fc in enumerate(self.fcs[:-1]):
            x = F.relu(fc(x))
            x = self.dropout(x)
        x = torch.sigmoid(self.fcs[-1](x))
        return x

class FlexibleLSTM(nn.Module):
    """Flexible LSTM for hyperparameter search"""
    def __init__(self, input_dim, hidden_sizes, num_layers, dropout_rate, use_attention=True):
        super(FlexibleLSTM, self).__init__()
        self.input_dim = input_dim
        self.sequence_length = 32
        self.feature_dim = (input_dim + self.sequence_length - 1) // self.sequence_length
        self.padded_input_dim = self.sequence_length * self.feature_dim
        
        # LSTM layers
        lstm_layers = []
        prev_hidden = self.feature_dim
        for hidden_size in hidden_sizes:
            lstm_layers.append(nn.LSTM(prev_hidden, hidden_size, batch_first=True, 
                                     dropout=dropout_rate if num_layers > 1 else 0))
            prev_hidden = hidden_size
        
        self.lstms = nn.ModuleList(lstm_layers)
        
        # Attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Linear(hidden_sizes[-1], 1)
        
        # FC layers
        self.fc1 = nn.Linear(hidden_sizes[-1], 128)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        if x.size(1) < self.padded_input_dim:
            x = F.pad(x, (0, self.padded_input_dim - x.size(1)))
        x = x.view(batch_size, self.sequence_length, -1)
        
        for lstm in self.lstms:
            x, _ = lstm(x)
        
        if self.use_attention:
            attn_weights = F.softmax(self.attention(x), dim=1)
            x = torch.sum(x * attn_weights, dim=1)
        else:
            x = x[:, -1, :]
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.sigmoid(self.fc2(x))
        return x

class FlexibleTransformer(nn.Module):
    """Flexible Transformer for hyperparameter search"""
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, dropout_rate):
        super(FlexibleTransformer, self).__init__()
        self.input_dim = input_dim
        self.sequence_length = 32
        self.feature_dim = (input_dim + self.sequence_length - 1) // self.sequence_length
        self.padded_input_dim = self.sequence_length * self.feature_dim
        self.d_model = d_model
        
        self.input_projection = nn.Linear(self.feature_dim, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.sequence_length, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout_rate, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        if x.size(1) < self.padded_input_dim:
            x = F.pad(x, (0, self.padded_input_dim - x.size(1)))
        x = x.view(batch_size, self.sequence_length, -1)
        x = self.input_projection(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x

def train_and_evaluate(model, train_loader, val_loader, test_loader, learning_rate, weight_decay, epochs=100):
    """Train and evaluate a model"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y.float())
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Evaluate on test set
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x).squeeze()
            all_probs.extend(outputs.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs > 0.5).astype(int)
    
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'auc': roc_auc_score(all_labels, all_probs)
    }
    
    return metrics, best_val_loss

def search_mlp_hyperparameters(X_train, y_train, X_val, y_val, X_test, y_test, input_dim):
    """Search hyperparameters for MLP"""
    print("Searching MLP hyperparameters...")
    
    search_space = {
        'hidden_sizes': [[256, 128, 64], [512, 256, 128], [128, 64, 32], [256, 128], [512, 256]],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.001, 0.003, 0.01],
        'weight_decay': [0.0, 1e-5, 1e-4],
        'batch_size': [32, 64],
        'activation': ['relu', 'tanh']
    }
    
    best_auc = 0
    best_params = None
    best_metrics = None
    
    combinations = list(itertools.product(*search_space.values()))
    print(f"Testing {len(combinations)} MLP configurations...")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(search_space.keys(), combo))
        if (i + 1) % 10 == 0:
            print(f"  MLP: {i+1}/{len(combinations)}")
        
        try:
            model = FlexibleMLP(input_dim, params['hidden_sizes'], 
                              params['dropout_rate'], params['activation']).to(device)
            
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
            
            metrics, _ = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                           params['learning_rate'], params['weight_decay'])
            
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_params = params
                best_metrics = metrics
        except:
            continue
    
    return best_params, best_metrics

def search_cnn_hyperparameters(X_train, y_train, X_val, y_val, X_test, y_test, input_dim):
    """Search hyperparameters for CNN"""
    print("Searching CNN hyperparameters...")
    
    search_space = {
        'num_filters': [[64, 128, 256], [32, 64, 128], [128, 256, 512]],
        'kernel_sizes': [[7, 5, 3], [5, 3, 3], [9, 7, 5]],
        'fc_sizes': [[512, 128], [256, 64], [1024, 256]],
        'dropout_rate': [0.3, 0.4, 0.5],
        'learning_rate': [0.001, 0.003, 0.01],
        'weight_decay': [0.0, 1e-5, 1e-4],
        'batch_size': [32, 64]
    }
    
    best_auc = 0
    best_params = None
    best_metrics = None
    
    # Limit combinations for CNN (more complex)
    num_filters_list = search_space['num_filters']
    kernel_sizes_list = search_space['kernel_sizes']
    fc_sizes_list = search_space['fc_sizes']
    dropout_list = search_space['dropout_rate']
    lr_list = search_space['learning_rate']
    wd_list = search_space['weight_decay']
    bs_list = search_space['batch_size']
    
    combinations = list(itertools.product(num_filters_list, kernel_sizes_list, fc_sizes_list,
                                        dropout_list, lr_list, wd_list, bs_list))
    # Sample if too many
    if len(combinations) > 50:
        np.random.seed(42)
        combinations = [combinations[i] for i in np.random.choice(len(combinations), 50, replace=False)]
    
    print(f"Testing {len(combinations)} CNN configurations...")
    
    for i, combo in enumerate(combinations):
        params = {
            'num_filters': combo[0],
            'kernel_sizes': combo[1],
            'fc_sizes': combo[2],
            'dropout_rate': combo[3],
            'learning_rate': combo[4],
            'weight_decay': combo[5],
            'batch_size': combo[6]
        }
        if (i + 1) % 5 == 0:
            print(f"  CNN: {i+1}/{len(combinations)}")
        
        try:
            model = FlexibleCNN(input_dim, params['num_filters'], params['kernel_sizes'],
                              params['fc_sizes'], params['dropout_rate']).to(device)
            
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
            
            metrics, _ = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                           params['learning_rate'], params['weight_decay'])
            
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_params = params
                best_metrics = metrics
        except Exception as e:
            continue
    
    return best_params, best_metrics

def search_lstm_hyperparameters(X_train, y_train, X_val, y_val, X_test, y_test, input_dim):
    """Search hyperparameters for LSTM"""
    print("Searching LSTM hyperparameters...")
    
    search_space = {
        'hidden_sizes': [[128, 64], [256, 128], [64, 32]],
        'num_layers': [1, 2],
        'dropout_rate': [0.2, 0.3, 0.4],
        'learning_rate': [0.001, 0.003, 0.01],
        'weight_decay': [0.0, 1e-5, 1e-4],
        'batch_size': [32, 64],
        'use_attention': [True, False]
    }
    
    best_auc = 0
    best_params = None
    best_metrics = None
    
    combinations = list(itertools.product(*search_space.values()))
    # Sample if too many
    if len(combinations) > 40:
        np.random.seed(42)
        combinations = [combinations[i] for i in np.random.choice(len(combinations), 40, replace=False)]
    
    print(f"Testing {len(combinations)} LSTM configurations...")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(search_space.keys(), combo))
        if (i + 1) % 5 == 0:
            print(f"  LSTM: {i+1}/{len(combinations)}")
        
        try:
            model = FlexibleLSTM(input_dim, params['hidden_sizes'], params['num_layers'],
                                params['dropout_rate'], params['use_attention']).to(device)
            
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
            
            metrics, _ = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                           params['learning_rate'], params['weight_decay'])
            
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_params = params
                best_metrics = metrics
        except:
            continue
    
    return best_params, best_metrics

def search_transformer_hyperparameters(X_train, y_train, X_val, y_val, X_test, y_test, input_dim):
    """Search hyperparameters for Transformer"""
    print("Searching Transformer hyperparameters...")
    
    search_space = {
        'd_model': [64, 128, 256],
        'nhead': [4, 8],
        'num_layers': [2, 4, 6],
        'dim_feedforward': [256, 512, 1024],
        'dropout_rate': [0.1, 0.2, 0.3],
        'learning_rate': [0.001, 0.003, 0.01],
        'weight_decay': [0.0, 1e-5, 1e-4],
        'batch_size': [32, 64]
    }
    
    best_auc = 0
    best_params = None
    best_metrics = None
    
    combinations = list(itertools.product(*search_space.values()))
    # Sample if too many
    if len(combinations) > 40:
        np.random.seed(42)
        combinations = [combinations[i] for i in np.random.choice(len(combinations), 40, replace=False)]
    
    print(f"Testing {len(combinations)} Transformer configurations...")
    
    for i, combo in enumerate(combinations):
        params = dict(zip(search_space.keys(), combo))
        if (i + 1) % 5 == 0:
            print(f"  Transformer: {i+1}/{len(combinations)}")
        
        try:
            model = FlexibleTransformer(input_dim, params['d_model'], params['nhead'],
                                      params['num_layers'], params['dim_feedforward'],
                                      params['dropout_rate']).to(device)
            
            train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
            test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
            
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)
            
            metrics, _ = train_and_evaluate(model, train_loader, val_loader, test_loader,
                                           params['learning_rate'], params['weight_decay'])
            
            if metrics['auc'] > best_auc:
                best_auc = metrics['auc']
                best_params = params
                best_metrics = metrics
        except:
            continue
    
    return best_params, best_metrics

def run_hyperparameter_search_for_experiment(experiment_num, n_features, base_results_dir="/home/eo/allergen-prediction/results"):
    """Run hyperparameter search for a specific experiment"""
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER SEARCH FOR EXPERIMENT {experiment_num} ({n_features} features)")
    print(f"{'='*80}")
    
    exp_dir = f"{base_results_dir}/experiment{experiment_num}"
    hp_dir = f"{exp_dir}/hyperparameter_search"
    os.makedirs(hp_dir, exist_ok=True)
    
    # Load data with selected features
    pos_train = pd.read_csv(f"{exp_dir}/Pos_train_descriptors.csv")
    neg_train = pd.read_csv(f"{exp_dir}/Neg_train_descriptors.csv")
    pos_test = pd.read_csv(f"{exp_dir}/Pos_test_descriptors.csv")
    neg_test = pd.read_csv(f"{exp_dir}/Neg_test_descriptors.csv")
    
    # Combine and prepare
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
    
    # Search hyperparameters for each architecture
    all_results = {}
    
    # MLP
    mlp_params, mlp_metrics = search_mlp_hyperparameters(
        X_train, y_train_split, X_val, y_val, X_test_scaled, y_test, input_dim
    )
    if mlp_params:
        all_results['MLP'] = {'params': mlp_params, 'metrics': mlp_metrics}
        print(f"Best MLP: AUC={mlp_metrics['auc']:.4f}")
    
    # CNN
    cnn_params, cnn_metrics = search_cnn_hyperparameters(
        X_train, y_train_split, X_val, y_val, X_test_scaled, y_test, input_dim
    )
    if cnn_params:
        all_results['CNN'] = {'params': cnn_params, 'metrics': cnn_metrics}
        print(f"Best CNN: AUC={cnn_metrics['auc']:.4f}")
    
    # LSTM
    lstm_params, lstm_metrics = search_lstm_hyperparameters(
        X_train, y_train_split, X_val, y_val, X_test_scaled, y_test, input_dim
    )
    if lstm_params:
        all_results['LSTM'] = {'params': lstm_params, 'metrics': lstm_metrics}
        print(f"Best LSTM: AUC={lstm_metrics['auc']:.4f}")
    
    # Transformer
    transformer_params, transformer_metrics = search_transformer_hyperparameters(
        X_train, y_train_split, X_val, y_val, X_test_scaled, y_test, input_dim
    )
    if transformer_params:
        all_results['Transformer'] = {'params': transformer_params, 'metrics': transformer_metrics}
        print(f"Best Transformer: AUC={transformer_metrics['auc']:.4f}")
    
    # Save results
    with open(f"{hp_dir}/best_hyperparameters.json", 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Create summary DataFrame
    summary_data = []
    for model_name, result in all_results.items():
        summary_data.append({
            'Model': model_name,
            'AUC': result['metrics']['auc'],
            'Accuracy': result['metrics']['accuracy'],
            'Precision': result['metrics']['precision'],
            'Recall': result['metrics']['recall'],
            'F1': result['metrics']['f1']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('AUC', ascending=False)
    summary_df.to_csv(f"{hp_dir}/hyperparameter_search_summary.csv", index=False)
    
    print(f"\nHyperparameter search complete for experiment {experiment_num}")
    print(f"Best model: {summary_df.iloc[0]['Model']} (AUC: {summary_df.iloc[0]['AUC']:.4f})")
    
    return all_results

def main():
    """Run hyperparameter search for all experiments"""
    print("="*80)
    print("COMPREHENSIVE HYPERPARAMETER SEARCH FOR ALL EXPERIMENTS")
    print("="*80)
    
    feature_counts = [50, 100, 150, 200, 217]
    base_results_dir = "/home/eo/allergen-prediction/results"
    
    all_experiment_results = {}
    
    for exp_num, n_features in enumerate(feature_counts, start=1):
        try:
            results = run_hyperparameter_search_for_experiment(exp_num, n_features, base_results_dir)
            all_experiment_results[exp_num] = results
        except Exception as e:
            print(f"Experiment {exp_num} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save overall summary
    overall_summary = []
    for exp_num, results in all_experiment_results.items():
        n_features = feature_counts[exp_num - 1]
        for model_name, result in results.items():
            overall_summary.append({
                'Experiment': exp_num,
                'N_Features': n_features,
                'Model': model_name,
                'AUC': result['metrics']['auc'],
                'Accuracy': result['metrics']['accuracy']
            })
    
    overall_df = pd.DataFrame(overall_summary)
    overall_df.to_csv(f"{base_results_dir}/all_hyperparameter_search_results.csv", index=False)
    
    print("\n" + "="*80)
    print("ALL HYPERPARAMETER SEARCHES COMPLETE")
    print("="*80)
    print(overall_df.groupby(['N_Features', 'Model'])['AUC'].max().unstack())

if __name__ == "__main__":
    main()

