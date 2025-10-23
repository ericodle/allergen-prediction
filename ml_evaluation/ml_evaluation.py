#!/usr/bin/env python3
"""
Comprehensive Machine Learning Evaluation for Allergen Prediction
This script evaluates multiple ML algorithms on allergen prediction data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                           roc_curve, precision_recall_curve, accuracy_score, 
                           precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AllergenMLEvaluator:
    def __init__(self, results_dir="/home/eo/allergen-prediction/results"):
        self.results_dir = results_dir
        self.eval_dir = f"{results_dir}/ml_evaluation"
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
    
    def initialize_models(self):
        """Initialize various ML models"""
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'AdaBoost': AdaBoostClassifier(random_state=42),
            'Neural Network': MLPClassifier(random_state=42, max_iter=1000)
        }
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        print("Training models...")
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all models and store results"""
        print("Evaluating models...")
        
        for name, model in self.models.items():
            print(f"Evaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
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
    
    def create_performance_plots(self, y_test):
        """Create comprehensive performance visualization plots"""
        print("Creating performance plots...")
        
        # 1. Model Comparison Bar Chart
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        model_names = list(self.results.keys())
        
        for i, metric in enumerate(metrics):
            ax = axes[i//3, i%3]
            values = [self.results[name][metric] for name in model_names]
            bars = ax.bar(model_names, values, alpha=0.7)
            ax.set_title(f'{metric.capitalize()} Comparison')
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
        plt.savefig(f'{self.eval_dir}/model_performance_comparison.png', dpi=300, bbox_inches='tight')
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
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.eval_dir}/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Precision-Recall Curves
        plt.figure(figsize=(10, 8))
        for name in model_names:
            precision, recall, _ = precision_recall_curve(y_test, self.results[name]['y_pred_proba'])
            plt.plot(recall, precision, label=f'{name}')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.eval_dir}/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Confusion Matrices
        n_models = len(model_names)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
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
    
    def create_feature_importance_plot(self, feature_names):
        """Create feature importance plot for tree-based models"""
        print("Creating feature importance plot...")
        
        # Get feature importance for tree-based models
        importance_models = ['Random Forest', 'Gradient Boosting', 'AdaBoost', 'Decision Tree']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, name in enumerate(importance_models):
            if name in self.models and hasattr(self.models[name], 'feature_importances_'):
                importances = self.models[name].feature_importances_
                indices = np.argsort(importances)[::-1][:20]  # Top 20 features
                
                ax = axes[i]
                ax.barh(range(len(indices)), importances[indices])
                ax.set_yticks(range(len(indices)))
                ax.set_yticklabels([feature_names[idx] for idx in indices])
                ax.set_title(f'{name} - Top 20 Feature Importances')
                ax.set_xlabel('Importance')
        
        plt.tight_layout()
        plt.savefig(f'{self.eval_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("Generating summary report...")
        
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
        results_df.to_csv(f'{self.eval_dir}/model_performance_summary.csv', index=False)
        
        # Print summary
        print("\n" + "="*80)
        print("MACHINE LEARNING EVALUATION SUMMARY")
        print("="*80)
        print(results_df.to_string(index=False))
        print("="*80)
        
        # Best model
        best_model = results_df.iloc[0]
        print(f"\nBest performing model: {best_model['Model']}")
        print(f"Best AUC score: {best_model['AUC']:.4f}")
        print(f"Best accuracy: {best_model['Accuracy']:.4f}")
        
        return results_df
    
    def run_evaluation(self):
        """Run the complete evaluation pipeline"""
        print("Starting comprehensive ML evaluation...")
        
        # Load data
        X_train, X_test, y_train, y_test, feature_names = self.load_data()
        
        # Initialize and train models
        self.initialize_models()
        self.train_models(X_train, y_train)
        
        # Evaluate models
        self.evaluate_models(X_test, y_test)
        
        # Create plots
        self.create_performance_plots(y_test)
        self.create_feature_importance_plot(feature_names)
        
        # Generate summary
        results_df = self.generate_summary_report()
        
        print(f"\nEvaluation complete! Results saved to: {self.eval_dir}")
        return results_df

if __name__ == "__main__":
    evaluator = AllergenMLEvaluator()
    results = evaluator.run_evaluation()
