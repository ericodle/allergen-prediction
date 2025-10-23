#!/usr/bin/env python3
"""
PyTorch Deep Learning Evaluation for Allergen Prediction
This script evaluates various deep learning architectures using PyTorch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, accuracy_score, 
                           precision_score, recall_score, f1_score)
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class BasicMLP(nn.Module):
    """Basic Multi-Layer Perceptron"""
    def __init__(self, input_dim):
        super(BasicMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(64, 1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        
        x = torch.sigmoid(self.fc4(x))
        return x

class CNN1D(nn.Module):
    """1D Convolutional Neural Network for sequence-like molecular descriptors"""
    def __init__(self, input_dim):
        super(CNN1D, self).__init__()
        # Reshape input to be sequence-like (batch_size, 1, input_dim)
        self.input_dim = input_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(2)
        
        # Calculate the size after convolutions
        conv_output_size = input_dim // 8 * 256  # After 3 pooling layers
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.dropout1 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, x):
        # Reshape to (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = torch.sigmoid(self.fc3(x))
        return x

class LSTM(nn.Module):
    """LSTM for sequence-like molecular descriptors"""
    def __init__(self, input_dim):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        
        # Reshape input to be sequence-like
        # We'll treat the molecular descriptors as a sequence
        self.sequence_length = 64  # Split input into sequences
        self.feature_dim = input_dim // self.sequence_length
        
        # Ensure we have the right feature dimension
        if input_dim % self.sequence_length != 0:
            self.padded_input_dim = ((input_dim + self.sequence_length - 1) // self.sequence_length) * self.sequence_length
            self.feature_dim = self.padded_input_dim // self.sequence_length
        else:
            self.padded_input_dim = input_dim
        
        # LSTM layers
        self.lstm1 = nn.LSTM(self.feature_dim, 128, batch_first=True, dropout=0.3)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True, dropout=0.2)
        
        # Attention mechanism
        self.attention = nn.Linear(64, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64, 128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 32)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape to sequence format
        # Pad or truncate to make it divisible by sequence_length
        if self.input_dim != self.padded_input_dim:
            pad_size = self.padded_input_dim - self.input_dim
            x = F.pad(x, (0, pad_size))
        
        x = x.view(batch_size, self.sequence_length, -1)
        
        # LSTM layers
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        
        # Attention mechanism
        attention_weights = F.softmax(self.attention(lstm_out2), dim=1)
        attended_output = torch.sum(lstm_out2 * attention_weights, dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(attended_output))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = torch.sigmoid(self.fc3(x))
        return x

class Transformer(nn.Module):
    """Transformer model for molecular descriptor analysis"""
    def __init__(self, input_dim):
        super(Transformer, self).__init__()
        self.input_dim = input_dim
        
        # Reshape input to be sequence-like
        self.sequence_length = 64  # Split input into sequences
        self.feature_dim = input_dim // self.sequence_length
        
        # Ensure we have the right feature dimension
        if input_dim % self.sequence_length != 0:
            self.padded_input_dim = ((input_dim + self.sequence_length - 1) // self.sequence_length) * self.sequence_length
            self.feature_dim = self.padded_input_dim // self.sequence_length
        else:
            self.padded_input_dim = input_dim
        
        self.d_model = 128
        
        # Input projection
        self.input_projection = nn.Linear(self.feature_dim, self.d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, self.sequence_length, self.d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape to sequence format
        if self.input_dim != self.padded_input_dim:
            pad_size = self.padded_input_dim - self.input_dim
            x = F.pad(x, (0, pad_size))
        
        x = x.view(batch_size, self.sequence_length, -1)
        
        # Project to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding
        
        # Transformer encoder
        x = self.transformer(x)
        
        # Global average pooling
        x = x.transpose(1, 2)  # (batch_size, d_model, sequence_length)
        x = self.global_pool(x)  # (batch_size, d_model, 1)
        x = x.squeeze(-1)  # (batch_size, d_model)
        
        # Classification
        x = self.classifier(x)
        return x

class AllergenPyTorchEvaluator:
    def __init__(self, results_dir="/home/eo/allergen-prediction/results"):
        self.results_dir = results_dir
        self.eval_dir = f"{results_dir}/deep_learning_evaluation"
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess the allergen data"""
        print("Loading data...")
        
        # Load training data
        pos_train = pd.read_csv(f"{self.results_dir}/Pos_train_descriptors.csv")
        neg_train = pd.read_csv(f"{self.results_dir}/Neg_train_descriptors.csv")
        
        # Load test data
        pos_test = pd.read_csv(f"{self.results_dir}/Pos_test_descriptors.csv")
        neg_test = pd.read_csv(f"{self.results_dir}/Neg_test_descriptors.csv")
        
        # Add labels
        pos_train['label'] = 1
        neg_train['label'] = 0
        pos_test['label'] = 1
        neg_test['label'] = 0
        
        # Combine datasets
        train_data = pd.concat([pos_train, neg_train], ignore_index=True)
        test_data = pd.concat([pos_test, neg_test], ignore_index=True)
        
        # Remove name column and separate features from labels
        X_train = train_data.drop(['Name', 'label'], axis=1)
        y_train = train_data['label']
        X_test = test_data.drop(['Name', 'label'], axis=1)
        y_test = test_data['label']
        
        # Handle missing values and infinite values
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set shape: {X_train_scaled.shape}")
        print(f"Test set shape: {X_test_scaled.shape}")
        print(f"Positive samples in training: {y_train.sum()}")
        print(f"Negative samples in training: {len(y_train) - y_train.sum()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train.columns
    
    def initialize_models(self, input_dim):
        """Initialize various PyTorch models"""
        self.models = {
            'Basic MLP': BasicMLP(input_dim).to(device),
            'CNN': CNN1D(input_dim).to(device),
            'LSTM': LSTM(input_dim).to(device),
            'Transformer': Transformer(input_dim).to(device)
        }
    
    def train_model(self, model, train_loader, val_loader, model_name, epochs=100):
        """Train a single PyTorch model"""
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y.float())
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                train_total += batch_y.size(0)
                train_correct += (predicted == batch_y).sum().item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y.float())
                    
                    val_loss += loss.item()
                    predicted = (outputs.squeeze() > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            # Calculate average losses and accuracies
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100 * train_correct / train_total
            val_acc = 100 * val_correct / val_total
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), f'{self.eval_dir}/best_{model_name.lower().replace(" ", "_")}.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Load best model
        model.load_state_dict(torch.load(f'{self.eval_dir}/best_{model_name.lower().replace(" ", "_")}.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train all PyTorch models"""
        print("Training PyTorch models...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train.values)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val.values)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        # Train each model
        model_names = list(self.models.keys())
        for name in model_names:
            print(f"Training {name}...")
            history = self.train_model(self.models[name], train_loader, val_loader, name)
            self.models[name + '_history'] = history
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and store results"""
        print("Evaluating PyTorch models...")
        
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        
        for name, model in self.models.items():
            if '_history' in name:
                continue
                
            print(f"Evaluating {name}...")
            
            model.eval()
            with torch.no_grad():
                y_pred_proba = model(X_test_tensor).cpu().numpy().flatten()
                y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Store results
            self.results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
    
    def create_training_plots(self):
        """Create training history plots"""
        print("Creating training history plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for i, (name, model) in enumerate(self.models.items()):
            if '_history' not in name:
                continue
                
            history = model
            model_name = name.replace('_history', '')
            
            # Loss plot
            ax = axes[0, 0]
            ax.plot(history['train_losses'], label=f'{model_name} - Train')
            ax.plot(history['val_losses'], label=f'{model_name} - Val')
            ax.set_title('Model Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Accuracy plot
            ax = axes[0, 1]
            ax.plot(history['train_accuracies'], label=f'{model_name} - Train')
            ax.plot(history['val_accuracies'], label=f'{model_name} - Val')
            ax.set_title('Model Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Accuracy (%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.eval_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_performance_plots(self, y_test):
        """Create performance visualization plots"""
        print("Creating deep learning performance plots...")
        
        # 1. Model Comparison Bar Chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        model_names = list(self.results.keys())
        
        for i, metric in enumerate(metrics):
            ax = axes[i//3, i%3]
            values = [self.results[name][metric] for name in model_names]
            bars = ax.bar(model_names, values, alpha=0.7)
            ax.set_title(f'{metric.capitalize()} Comparison (Deep Learning)', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.capitalize())
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
        
        # Hide the last subplot if not needed
        if len(metrics) < 6:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.eval_dir}/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curves
        plt.figure(figsize=(10, 8))
        for name in model_names:
            fpr, tpr, _ = roc_curve(y_test, self.results[name]['y_pred_proba'])
            auc = self.results[name]['auc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison (Deep Learning)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.eval_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion Matrices
        n_models = len(model_names)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, name in enumerate(model_names):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            cm = confusion_matrix(y_test, self.results[name]['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f'{name}\nAccuracy: {self.results[name]["accuracy"]:.3f}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.eval_dir}/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("Generating deep learning summary report...")
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[name]['accuracy'] for name in self.results.keys()],
            'Precision': [self.results[name]['precision'] for name in self.results.keys()],
            'Recall': [self.results[name]['recall'] for name in self.results.keys()],
            'F1-Score': [self.results[name]['f1'] for name in self.results.keys()],
            'AUC': [self.results[name]['auc'] for name in self.results.keys()]
        })
        
        # Sort by AUC score
        results_df = results_df.sort_values('AUC', ascending=False)
        
        # Save to CSV
        results_df.to_csv(f'{self.eval_dir}/performance_summary.csv', index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("DEEP LEARNING EVALUATION SUMMARY")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        
        # Best model
        best_model = results_df.iloc[0]
        print(f"\nBest performing deep learning model: {best_model['Model']}")
        print(f"Best AUC score: {best_model['AUC']:.4f}")
        print(f"Best accuracy: {best_model['Accuracy']:.4f}")
        
        return results_df
    
    def run_evaluation(self):
        """Run the complete deep learning evaluation pipeline"""
        print("Starting comprehensive deep learning evaluation...")
        
        # Load data
        X_train, X_test, y_train, y_test, feature_names = self.load_data()
        
        # Split training data for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Initialize and train models
        self.initialize_models(X_train.shape[1])
        self.train_models(X_train_split, y_train_split, X_val, y_val)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Create plots
        self.create_training_plots()
        self.create_performance_plots(y_test)
        
        # Generate summary
        results_df = self.generate_summary_report()
        
        print(f"\nDeep learning evaluation complete! Results saved to: {self.eval_dir}")
        return results_df

if __name__ == "__main__":
    evaluator = AllergenPyTorchEvaluator()
    results = evaluator.run_evaluation()
