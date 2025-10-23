#!/usr/bin/env python3
"""
Comprehensive Comparison of ML and Deep Learning Models for Allergen Prediction
This script combines and compares results from both traditional ML and deep learning models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ComprehensiveComparison:
    def __init__(self, results_dir="/home/eo/allergen-prediction/results"):
        self.results_dir = results_dir
        self.eval_dir = f"{results_dir}/ml_evaluation"
        
    def load_results(self):
        """Load results from both ML and deep learning evaluations"""
        print("Loading results from both evaluations...")
        
        # Load traditional ML results
        ml_results = pd.read_csv(f"{self.results_dir}/ml_evaluation/model_performance_summary.csv")
        ml_results['Model_Type'] = 'Traditional ML'
        
        # Load deep learning results
        dl_results = pd.read_csv(f"{self.results_dir}/deep_learning_evaluation/performance_summary.csv")
        dl_results['Model_Type'] = 'Deep Learning'
        
        # Combine results
        combined_results = pd.concat([ml_results, dl_results], ignore_index=True)
        
        print(f"Loaded {len(ml_results)} traditional ML models")
        print(f"Loaded {len(dl_results)} deep learning models")
        print(f"Total models: {len(combined_results)}")
        
        return combined_results, ml_results, dl_results
    
    def create_comprehensive_plots(self, combined_results, ml_results, dl_results):
        """Create comprehensive comparison plots"""
        print("Creating comprehensive comparison plots...")
        
        # 1. Traditional ML Performance Comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        model_names = list(ml_results['Model'])
        
        for i, metric in enumerate(metrics):
            ax = axes[i//3, i%3]
            values = ml_results[metric].values
            bars = ax.bar(model_names, values, alpha=0.7, color='#1f77b4')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'{metric} Comparison (Traditional ML)', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Hide the last subplot if not needed
        if len(metrics) < 6:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/ml_evaluation/traditional_ml_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Deep Learning Performance Comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        model_names = list(dl_results['Model'])
        
        for i, metric in enumerate(metrics):
            ax = axes[i//3, i%3]
            values = dl_results[metric].values
            bars = ax.bar(model_names, values, alpha=0.7, color='#ff7f0e')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=10)
            
            ax.set_title(f'{metric} Comparison (Deep Learning)', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Hide the last subplot if not needed
        if len(metrics) < 6:
            axes[1, 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/deep_learning_evaluation/deep_learning_performance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Model Performance Scatter Plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create scatter plot with different colors for each model type
        colors = {'Traditional ML': '#1f77b4', 'Deep Learning': '#ff7f0e'}
        
        for model_type in combined_results['Model_Type'].unique():
            type_data = combined_results[combined_results['Model_Type'] == model_type]
            ax.scatter(type_data['Accuracy'], type_data['AUC'], 
                      c=colors[model_type], label=model_type, s=100, alpha=0.7)
            
            # Add model names as annotations
            for _, row in type_data.iterrows():
                ax.annotate(row['Model'], (row['Accuracy'], row['AUC']), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Accuracy')
        ax.set_ylabel('AUC Score')
        ax.set_title('Model Performance: Accuracy vs AUC Score', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/performance_scatter.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, combined_results, ml_results, dl_results):
        """Generate a comprehensive analysis report"""
        print("Generating comprehensive analysis report...")
        
        # Sort by AUC score
        combined_results_sorted = combined_results.sort_values('AUC', ascending=False)
        
        # Save combined results
        combined_results_sorted.to_csv(f'{self.results_dir}/comprehensive_results_summary.csv', 
                                     index=False)
        
        # Generate statistics
        print("\n" + "="*100)
        print("COMPREHENSIVE MACHINE LEARNING EVALUATION REPORT")
        print("="*100)
        
        print(f"\nDataset Overview:")
        print(f"- Total models evaluated: {len(combined_results)}")
        print(f"- Traditional ML models: {len(ml_results)}")
        print(f"- Deep Learning models: {len(dl_results)}")
        
        print(f"\nBest Overall Model:")
        best_model = combined_results_sorted.iloc[0]
        print(f"- Model: {best_model['Model']} ({best_model['Model_Type']})")
        print(f"- AUC Score: {best_model['AUC']:.4f}")
        print(f"- Accuracy: {best_model['Accuracy']:.4f}")
        print(f"- Precision: {best_model['Precision']:.4f}")
        print(f"- Recall: {best_model['Recall']:.4f}")
        print(f"- F1-Score: {best_model['F1-Score']:.4f}")
        
        print(f"\nTop 5 Models by AUC Score:")
        print("-" * 80)
        for i, (_, model) in enumerate(combined_results_sorted.head().iterrows()):
            print(f"{i+1}. {model['Model']} ({model['Model_Type']}) - AUC: {model['AUC']:.4f}")
        
        print(f"\nModel Type Performance Summary:")
        print("-" * 50)
        type_stats = combined_results.groupby('Model_Type').agg({
            'Accuracy': ['mean', 'std', 'max'],
            'Precision': ['mean', 'std', 'max'],
            'Recall': ['mean', 'std', 'max'],
            'F1-Score': ['mean', 'std', 'max'],
            'AUC': ['mean', 'std', 'max']
        }).round(4)
        
        print(type_stats)
        
        print(f"\nBest Traditional ML Model:")
        best_ml = ml_results.loc[ml_results['AUC'].idxmax()]
        print(f"- Model: {best_ml['Model']}")
        print(f"- AUC Score: {best_ml['AUC']:.4f}")
        print(f"- Accuracy: {best_ml['Accuracy']:.4f}")
        
        print(f"\nBest Deep Learning Model:")
        best_dl = dl_results.loc[dl_results['AUC'].idxmax()]
        print(f"- Model: {best_dl['Model']}")
        print(f"- AUC Score: {best_dl['AUC']:.4f}")
        print(f"- Accuracy: {best_dl['Accuracy']:.4f}")
        
        # Performance comparison
        ml_avg_auc = ml_results['AUC'].mean()
        dl_avg_auc = dl_results['AUC'].mean()
        
        print(f"\nAverage Performance Comparison:")
        print(f"- Traditional ML Average AUC: {ml_avg_auc:.4f}")
        print(f"- Deep Learning Average AUC: {dl_avg_auc:.4f}")
        
        if ml_avg_auc > dl_avg_auc:
            print(f"- Traditional ML models perform better on average")
        else:
            print(f"- Deep Learning models perform better on average")
        
        print("="*100)
        
        return combined_results_sorted
    
    def run_comprehensive_analysis(self):
        """Run the complete comprehensive analysis"""
        print("Starting comprehensive analysis...")
        
        # Load results
        combined_results, ml_results, dl_results = self.load_results()
        
        # Create plots
        self.create_comprehensive_plots(combined_results, ml_results, dl_results)
        
        # Generate report
        final_results = self.generate_comprehensive_report(combined_results, ml_results, dl_results)
        
        print(f"\nComprehensive analysis complete! Results saved to: {self.results_dir}")
        return final_results

if __name__ == "__main__":
    analyzer = ComprehensiveComparison()
    results = analyzer.run_comprehensive_analysis()
