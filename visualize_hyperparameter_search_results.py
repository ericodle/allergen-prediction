#!/usr/bin/env python3
"""
Visualize Hyperparameter Search Results (Phase 1)
Creates comprehensive figures showing model performance across feature counts
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

def load_hyperparameter_search_results(base_dir="/home/eo/allergen-prediction/results"):
    """Load all hyperparameter search results from experiment directories"""
    feature_counts = [50, 100, 150, 200, 217]
    all_results = []
    
    for exp_num, n_features in enumerate(feature_counts, start=1):
        hp_summary_file = f"{base_dir}/experiment{exp_num}/hyperparameter_search/hyperparameter_search_summary.csv"
        
        if os.path.exists(hp_summary_file):
            df = pd.read_csv(hp_summary_file)
            df['Experiment'] = exp_num
            df['N_Features'] = n_features
            all_results.append(df)
        else:
            print(f"Warning: {hp_summary_file} not found")
    
    if not all_results:
        return None
    
    results_df = pd.concat(all_results, ignore_index=True)
    return results_df

def create_hyperparameter_search_visualizations(results_df, output_dir="/home/eo/allergen-prediction/results"):
    """Create comprehensive visualizations of hyperparameter search results"""
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = f"{output_dir}/phase1_hyperparameter_search_figures"
    os.makedirs(figures_dir, exist_ok=True)
    
    # Model name mapping for display
    model_display_names = {
        'MLP': 'MLP',
        'CNN': 'CNN',
        'LSTM': 'LSTM',
        'Transformer': 'Transformer'
    }
    
    # 1. AUC Comparison Across Feature Counts
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        ax.plot(model_data['N_Features'], model_data['AUC'], 
                marker='o', linewidth=2, markersize=8, label=model_display_names.get(model, model))
    
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_title('Hyperparameter Search Results: AUC-ROC Across Feature Counts', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([50, 100, 150, 200, 217])
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/auc_across_features.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir}/auc_across_features.png")
    
    # 2. Accuracy Comparison Across Feature Counts
    fig, ax = plt.subplots(figsize=(12, 6))
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        ax.plot(model_data['N_Features'], model_data['Accuracy'], 
                marker='s', linewidth=2, markersize=8, label=model_display_names.get(model, model))
    
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Hyperparameter Search Results: Accuracy Across Feature Counts', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([50, 100, 150, 200, 217])
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/accuracy_across_features.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir}/accuracy_across_features.png")
    
    # 3. Heatmap of AUC by Model and Feature Count
    pivot_auc = results_df.pivot(index='Model', columns='N_Features', values='AUC')
    pivot_auc = pivot_auc.reindex(['MLP', 'CNN', 'LSTM', 'Transformer'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_auc, annot=True, fmt='.3f', cmap='YlOrRd', 
                cbar_kws={'label': 'AUC-ROC'}, ax=ax, vmin=0.85, vmax=0.95)
    ax.set_title('Hyperparameter Search Results: AUC-ROC Heatmap', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/auc_heatmap.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir}/auc_heatmap.png")
    
    # 4. Heatmap of Accuracy by Model and Feature Count
    pivot_acc = results_df.pivot(index='Model', columns='N_Features', values='Accuracy')
    pivot_acc = pivot_acc.reindex(['MLP', 'CNN', 'LSTM', 'Transformer'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot_acc, annot=True, fmt='.3f', cmap='YlGnBu', 
                cbar_kws={'label': 'Accuracy'}, ax=ax, vmin=0.80, vmax=0.90)
    ax.set_title('Hyperparameter Search Results: Accuracy Heatmap', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/accuracy_heatmap.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir}/accuracy_heatmap.png")
    
    # 5. Bar plot comparing best model per feature count
    best_models = results_df.loc[results_df.groupby('N_Features')['AUC'].idxmax()]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = np.arange(len(best_models))
    bars = ax.bar(x_pos, best_models['AUC'], 
                  color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Best AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_title('Best Model Performance by Feature Count (Hyperparameter Search)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{n} ({m})" for n, m in zip(best_models['N_Features'], best_models['Model'])])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, auc, model) in enumerate(zip(bars, best_models['AUC'], best_models['Model'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{auc:.3f}\n({model})',
                ha='center', va='bottom', fontsize=9)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/best_model_by_features.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir}/best_model_by_features.png")
    
    # 6. Summary statistics table
    summary_stats = results_df.groupby('Model').agg({
        'AUC': ['mean', 'std', 'min', 'max'],
        'Accuracy': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary_stats.to_csv(f"{figures_dir}/model_summary_statistics.csv")
    print(f"Saved: {figures_dir}/model_summary_statistics.csv")
    
    # 7. Comprehensive comparison plot (AUC and Accuracy side by side)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # AUC plot
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        ax1.plot(model_data['N_Features'], model_data['AUC'], 
                marker='o', linewidth=2, markersize=8, label=model_display_names.get(model, model))
    ax1.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax1.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax1.set_title('AUC-ROC Across Feature Counts', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([50, 100, 150, 200, 217])
    
    # Accuracy plot
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        ax2.plot(model_data['N_Features'], model_data['Accuracy'], 
                marker='s', linewidth=2, markersize=8, label=model_display_names.get(model, model))
    ax2.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax2.set_title('Accuracy Across Feature Counts', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([50, 100, 150, 200, 217])
    
    plt.suptitle('Hyperparameter Search Results: Comprehensive Model Comparison', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{figures_dir}/comprehensive_comparison.png", bbox_inches='tight')
    plt.close()
    print(f"Saved: {figures_dir}/comprehensive_comparison.png")
    
    print(f"\nAll visualizations saved to: {figures_dir}/")
    return figures_dir

def main():
    """Main function to create visualizations"""
    print("="*80)
    print("VISUALIZING HYPERPARAMETER SEARCH RESULTS (PHASE 1)")
    print("="*80)
    
    # Load results
    results_df = load_hyperparameter_search_results()
    
    if results_df is None:
        print("Error: No hyperparameter search results found!")
        return
    
    print(f"\nLoaded results for {len(results_df)} model configurations")
    print(f"Models: {', '.join(results_df['Model'].unique())}")
    print(f"Feature counts: {sorted(results_df['N_Features'].unique())}")
    
    # Create visualizations
    figures_dir = create_hyperparameter_search_visualizations(results_df)
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    summary = results_df.groupby('Model').agg({
        'AUC': ['mean', 'std', 'max'],
        'Accuracy': ['mean', 'std', 'max']
    }).round(4)
    print(summary)
    
    print("\n" + "="*80)
    print("BEST MODEL BY FEATURE COUNT")
    print("="*80)
    best_by_features = results_df.loc[results_df.groupby('N_Features')['AUC'].idxmax()]
    for _, row in best_by_features.iterrows():
        print(f"{row['N_Features']} features: {row['Model']} (AUC={row['AUC']:.4f}, Acc={row['Accuracy']:.4f})")

if __name__ == "__main__":
    main()

