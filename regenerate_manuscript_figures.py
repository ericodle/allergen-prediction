#!/usr/bin/env python3
"""
Regenerate manuscript figures with Phase 2 results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def load_phase2_results():
    """Load Phase 2 comparative evaluation results"""
    base_dir = "/home/eo/allergen-prediction/results"
    phase2_file = f"{base_dir}/phase2_comparative_evaluation/overall_summary.csv"
    
    if os.path.exists(phase2_file):
        df = pd.read_csv(phase2_file)
        df['Model_Type'] = 'Deep Learning'
        return df
    return None

def load_ml_results():
    """Try to load ML results from experiment directories"""
    base_dir = "/home/eo/allergen-prediction/results"
    all_ml = []
    
    for exp_num in range(1, 6):
        ml_file = f"{base_dir}/experiment{exp_num}/ml_evaluation/model_performance_summary.csv"
        if os.path.exists(ml_file):
            df = pd.read_csv(ml_file)
            df['Experiment'] = exp_num
            df['Model_Type'] = 'Traditional ML'
            df['N_Features'] = [50, 100, 150, 200, 217][exp_num - 1]
            all_ml.append(df)
    
    if all_ml:
        return pd.concat(all_ml, ignore_index=True)
    return None

def create_feature_count_comparison(ml_df, dl_df, output_dir):
    """Create feature count comparison figure"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Define colors for DL models
    dl_colors = {'MLP': '#ff7f0e', 'CNN': '#2ca02c', 'LSTM': '#d62728', 'Transformer': '#9467bd'}
    dl_markers = {'MLP': 'o', 'CNN': 's', 'LSTM': '^', 'Transformer': 'D'}
    
    # Plot 1: Best AUC by feature count
    ax = axes[0]
    if ml_df is not None:
        best_ml = ml_df.groupby('N_Features')['AUC'].max()
        ax.plot(best_ml.index, best_ml.values, 'o-', label='Traditional ML (Best)', linewidth=2, markersize=8, color='#1f77b4')
    if dl_df is not None:
        # Plot each DL model separately
        for model in ['MLP', 'CNN', 'LSTM', 'Transformer']:
            model_data = dl_df[dl_df['Model'] == model]
            if len(model_data) > 0:
                model_auc = model_data.groupby('N_Features')['AUC'].mean()
                ax.plot(model_auc.index, model_auc.values, f'{dl_markers[model]}-', 
                       label=model, linewidth=2, markersize=6, color=dl_colors[model], alpha=0.8)
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('AUC Score vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average AUC by feature count
    ax = axes[1]
    if ml_df is not None:
        avg_ml = ml_df.groupby('N_Features')['AUC'].mean()
        ax.plot(avg_ml.index, avg_ml.values, 'o-', label='Traditional ML (Avg)', linewidth=2, markersize=8, color='#1f77b4')
    if dl_df is not None:
        # Plot each DL model separately
        for model in ['MLP', 'CNN', 'LSTM', 'Transformer']:
            model_data = dl_df[dl_df['Model'] == model]
            if len(model_data) > 0:
                model_auc = model_data.groupby('N_Features')['AUC'].mean()
                ax.plot(model_auc.index, model_auc.values, f'{dl_markers[model]}-', 
                       label=model, linewidth=2, markersize=6, color=dl_colors[model], alpha=0.8)
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Average AUC Score vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Best Accuracy by feature count
    ax = axes[2]
    if ml_df is not None:
        best_acc_ml = ml_df.groupby('N_Features')['Accuracy'].max()
        ax.plot(best_acc_ml.index, best_acc_ml.values, 'o-', label='Traditional ML (Best)', linewidth=2, markersize=8, color='#1f77b4')
    if dl_df is not None:
        # Plot each DL model separately
        for model in ['MLP', 'CNN', 'LSTM', 'Transformer']:
            model_data = dl_df[dl_df['Model'] == model]
            if len(model_data) > 0:
                model_acc = model_data.groupby('N_Features')['Accuracy'].mean()
                ax.plot(model_acc.index, model_acc.values, f'{dl_markers[model]}-', 
                       label=model, linewidth=2, markersize=6, color=dl_colors[model], alpha=0.8)
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_count_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/feature_count_comparison.png")

def create_accuracy_summary(ml_df, dl_df, output_dir):
    """Create accuracy summary figure"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Define colors for DL models
    dl_colors = {'MLP': '#ff7f0e', 'CNN': '#2ca02c', 'LSTM': '#d62728', 'Transformer': '#9467bd'}
    
    # Plot 1: Best Accuracy by feature count
    ax = axes[0, 0]
    if ml_df is not None:
        best_acc_ml = ml_df.groupby('N_Features')['Accuracy'].max()
        ax.plot(best_acc_ml.index, best_acc_ml.values, 'o-', label='Traditional ML (Best)', linewidth=2, markersize=8, color='#1f77b4')
    if dl_df is not None:
        # Plot each DL model separately
        for model in ['MLP', 'CNN', 'LSTM', 'Transformer']:
            model_data = dl_df[dl_df['Model'] == model]
            if len(model_data) > 0:
                model_acc = model_data.groupby('N_Features')['Accuracy'].mean()
                ax.plot(model_acc.index, model_acc.values, 's-', 
                       label=model, linewidth=2, markersize=6, color=dl_colors[model], alpha=0.8)
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Best Accuracy vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.0])
    
    # Plot 2: Average Accuracy by feature count
    ax = axes[0, 1]
    if ml_df is not None:
        avg_acc_ml = ml_df.groupby('N_Features')['Accuracy'].mean()
        ax.plot(avg_acc_ml.index, avg_acc_ml.values, 'o-', label='Traditional ML (Avg)', linewidth=2, markersize=8, color='#1f77b4')
    if dl_df is not None:
        # Plot each DL model separately
        for model in ['MLP', 'CNN', 'LSTM', 'Transformer']:
            model_data = dl_df[dl_df['Model'] == model]
            if len(model_data) > 0:
                model_acc = model_data.groupby('N_Features')['Accuracy'].mean()
                ax.plot(model_acc.index, model_acc.values, 's-', 
                       label=model, linewidth=2, markersize=6, color=dl_colors[model], alpha=0.8)
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Average Accuracy vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.0])
    
    # Plot 3: Accuracy distribution (boxplot)
    ax = axes[1, 0]
    if ml_df is not None and dl_df is not None:
        combined = pd.concat([ml_df[['N_Features', 'Accuracy', 'Model_Type']], 
                             dl_df[['N_Features', 'Accuracy', 'Model_Type']]], ignore_index=True)
        sns.boxplot(data=combined, x='N_Features', y='Accuracy', hue='Model_Type', ax=ax)
    elif dl_df is not None:
        sns.boxplot(data=dl_df, x='N_Features', y='Accuracy', ax=ax, color='#ff7f0e')
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy Distribution by Feature Count', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Accuracy vs AUC scatter
    ax = axes[1, 1]
    if ml_df is not None:
        ax.scatter(ml_df['AUC'], ml_df['Accuracy'], label='Traditional ML', alpha=0.6, s=50, color='#1f77b4')
    if dl_df is not None:
        # Plot each DL model separately
        for model in ['MLP', 'CNN', 'LSTM', 'Transformer']:
            model_data = dl_df[dl_df['Model'] == model]
            if len(model_data) > 0:
                ax.scatter(model_data['AUC'], model_data['Accuracy'], 
                          label=model, alpha=0.6, s=50, color=dl_colors[model])
    ax.set_xlabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy vs AUC-ROC', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/accuracy_summary.png")

def main():
    """Main function"""
    print("Regenerating manuscript figures with Phase 2 results...")
    
    # Load data
    dl_df = load_phase2_results()
    ml_df = load_ml_results()
    
    if dl_df is None:
        print("Error: Phase 2 results not found!")
        return
    
    print(f"Loaded Phase 2 DL results: {len(dl_df)} rows")
    if ml_df is not None:
        print(f"Loaded ML results: {len(ml_df)} rows")
    else:
        print("Warning: ML results not found - figures will only show DL results")
    
    # Create output directory
    output_dir = "/home/eo/allergen-prediction/manuscript/figures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate figures
    create_feature_count_comparison(ml_df, dl_df, output_dir)
    create_accuracy_summary(ml_df, dl_df, output_dir)
    
    print("\nFigures regenerated successfully!")

if __name__ == "__main__":
    main()

