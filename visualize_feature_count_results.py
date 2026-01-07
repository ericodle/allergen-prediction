#!/usr/bin/env python3
"""
Visualize and compare results across different feature counts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_all_results(base_dir="/home/eo/allergen-prediction/results"):
    """Load results from all experiments"""
    results = []
    
    for exp_num in range(1, 6):
        exp_dir = f"{base_dir}/experiment{exp_num}"
        
        # Load ML results
        ml_file = f"{exp_dir}/ml_evaluation/model_performance_summary.csv"
        dl_file = f"{exp_dir}/deep_learning_evaluation/performance_summary.csv"
        
        if os.path.exists(ml_file):
            ml_df = pd.read_csv(ml_file)
            ml_df['Experiment'] = exp_num
            ml_df['Model_Type'] = 'Traditional ML'
            ml_df['N_Features'] = [50, 100, 150, 200, 217][exp_num - 1]
            results.append(ml_df)
        
        if os.path.exists(dl_file):
            dl_df = pd.read_csv(dl_file)
            dl_df['Experiment'] = exp_num
            dl_df['Model_Type'] = 'Deep Learning'
            dl_df['N_Features'] = [50, 100, 150, 200, 217][exp_num - 1]
            results.append(dl_df)
    
    if results:
        return pd.concat(results, ignore_index=True)
    return None

def create_comparison_plots(results_df, output_dir="/home/eo/allergen-prediction/results"):
    """Create comparison plots"""
    
    # 1. AUC vs Feature Count
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Best AUC by feature count
    ax = axes[0, 0]
    best_ml = results_df[results_df['Model_Type'] == 'Traditional ML'].groupby('N_Features')['AUC'].max()
    best_dl = results_df[results_df['Model_Type'] == 'Deep Learning'].groupby('N_Features')['AUC'].max()
    
    ax.plot(best_ml.index, best_ml.values, 'o-', label='Traditional ML', linewidth=2, markersize=8)
    ax.plot(best_dl.index, best_dl.values, 's-', label='Deep Learning', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Best AUC Score', fontsize=12)
    ax.set_title('Best AUC Score vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Average AUC by feature count
    ax = axes[0, 1]
    avg_ml = results_df[results_df['Model_Type'] == 'Traditional ML'].groupby('N_Features')['AUC'].mean()
    avg_dl = results_df[results_df['Model_Type'] == 'Deep Learning'].groupby('N_Features')['AUC'].mean()
    
    ax.plot(avg_ml.index, avg_ml.values, 'o-', label='Traditional ML', linewidth=2, markersize=8)
    ax.plot(avg_dl.index, avg_dl.values, 's-', label='Deep Learning', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Average AUC Score', fontsize=12)
    ax.set_title('Average AUC Score vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Best Accuracy by feature count
    ax = axes[1, 0]
    best_acc_ml = results_df[results_df['Model_Type'] == 'Traditional ML'].groupby('N_Features')['Accuracy'].max()
    best_acc_dl = results_df[results_df['Model_Type'] == 'Deep Learning'].groupby('N_Features')['Accuracy'].max()
    
    ax.plot(best_acc_ml.index, best_acc_ml.values, 'o-', label='Traditional ML', linewidth=2, markersize=8)
    ax.plot(best_acc_dl.index, best_acc_dl.values, 's-', label='Deep Learning', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Best Accuracy', fontsize=12)
    ax.set_title('Best Accuracy vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Model count comparison
    ax = axes[1, 1]
    ml_counts = results_df[results_df['Model_Type'] == 'Traditional ML'].groupby('N_Features').size()
    dl_counts = results_df[results_df['Model_Type'] == 'Deep Learning'].groupby('N_Features').size()
    
    x = np.arange(len(ml_counts.index))
    width = 0.35
    ax.bar(x - width/2, ml_counts.values, width, label='Traditional ML', alpha=0.7)
    ax.bar(x + width/2, dl_counts.values, width, label='Deep Learning', alpha=0.7)
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Number of Models', fontsize=12)
    ax.set_title('Number of Models Evaluated', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(ml_counts.index)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_count_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed model performance heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create pivot table for AUC scores
    ml_data = results_df[results_df['Model_Type'] == 'Traditional ML']
    pivot_ml = ml_data.pivot_table(values='AUC', index='Model', columns='N_Features', aggfunc='mean')
    
    sns.heatmap(pivot_ml, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'AUC Score'})
    ax.set_title('Traditional ML Models: AUC Scores by Feature Count', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ml_models_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Deep Learning models AUC heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dl_data = results_df[results_df['Model_Type'] == 'Deep Learning']
    pivot_dl = dl_data.pivot_table(values='AUC', index='Model', columns='N_Features', aggfunc='mean')
    
    sns.heatmap(pivot_dl, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'AUC Score'})
    ax.set_title('Deep Learning Models: AUC Scores by Feature Count', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dl_models_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Traditional ML models Accuracy heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    
    ml_data = results_df[results_df['Model_Type'] == 'Traditional ML']
    pivot_ml_acc = ml_data.pivot_table(values='Accuracy', index='Model', columns='N_Features', aggfunc='mean')
    
    sns.heatmap(pivot_ml_acc, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Accuracy'})
    ax.set_title('Traditional ML Models: Accuracy by Feature Count', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ml_models_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Deep Learning models Accuracy heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dl_data = results_df[results_df['Model_Type'] == 'Deep Learning']
    pivot_dl_acc = dl_data.pivot_table(values='Accuracy', index='Model', columns='N_Features', aggfunc='mean')
    
    sns.heatmap(pivot_dl_acc, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Accuracy'})
    ax.set_title('Deep Learning Models: Accuracy by Feature Count', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/dl_models_accuracy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Comprehensive Accuracy Summary Figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Best Accuracy by feature count
    ax = axes[0, 0]
    best_acc_ml = results_df[results_df['Model_Type'] == 'Traditional ML'].groupby('N_Features')['Accuracy'].max()
    best_acc_dl = results_df[results_df['Model_Type'] == 'Deep Learning'].groupby('N_Features')['Accuracy'].max()
    
    ax.plot(best_acc_ml.index, best_acc_ml.values, 'o-', label='Traditional ML', linewidth=2, markersize=8, color='#1f77b4')
    ax.plot(best_acc_dl.index, best_acc_dl.values, 's-', label='Deep Learning', linewidth=2, markersize=8, color='#ff7f0e')
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Best Accuracy', fontsize=12)
    ax.set_title('Best Accuracy vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.0])
    
    # Plot 2: Average Accuracy by feature count
    ax = axes[0, 1]
    avg_acc_ml = results_df[results_df['Model_Type'] == 'Traditional ML'].groupby('N_Features')['Accuracy'].mean()
    avg_acc_dl = results_df[results_df['Model_Type'] == 'Deep Learning'].groupby('N_Features')['Accuracy'].mean()
    
    ax.plot(avg_acc_ml.index, avg_acc_ml.values, 'o-', label='Traditional ML', linewidth=2, markersize=8, color='#1f77b4')
    ax.plot(avg_acc_dl.index, avg_acc_dl.values, 's-', label='Deep Learning', linewidth=2, markersize=8, color='#ff7f0e')
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Average Accuracy vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.7, 1.0])
    
    # Plot 3: Accuracy distribution boxplot
    ax = axes[1, 0]
    ml_acc_data = []
    dl_acc_data = []
    feature_counts = sorted(results_df['N_Features'].unique())
    
    for n_feat in feature_counts:
        ml_subset = results_df[(results_df['Model_Type'] == 'Traditional ML') & (results_df['N_Features'] == n_feat)]
        dl_subset = results_df[(results_df['Model_Type'] == 'Deep Learning') & (results_df['N_Features'] == n_feat)]
        ml_acc_data.append(ml_subset['Accuracy'].values)
        dl_acc_data.append(dl_subset['Accuracy'].values)
    
    positions_ml = [x - 0.2 for x in range(len(feature_counts))]
    positions_dl = [x + 0.2 for x in range(len(feature_counts))]
    
    bp1 = ax.boxplot(ml_acc_data, positions=positions_ml, widths=0.35, patch_artist=True,
                     boxprops=dict(facecolor='#1f77b4', alpha=0.7),
                     medianprops=dict(color='black', linewidth=2))
    bp2 = ax.boxplot(dl_acc_data, positions=positions_dl, widths=0.35, patch_artist=True,
                     boxprops=dict(facecolor='#ff7f0e', alpha=0.7),
                     medianprops=dict(color='black', linewidth=2))
    
    ax.set_xticks(range(len(feature_counts)))
    ax.set_xticklabels(feature_counts)
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy Distribution by Feature Count', fontsize=14, fontweight='bold')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['Traditional ML', 'Deep Learning'], fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Accuracy vs AUC scatter
    ax = axes[1, 1]
    ml_subset = results_df[results_df['Model_Type'] == 'Traditional ML']
    dl_subset = results_df[results_df['Model_Type'] == 'Deep Learning']
    
    for n_feat in feature_counts:
        ml_data = ml_subset[ml_subset['N_Features'] == n_feat]
        dl_data = dl_subset[dl_subset['N_Features'] == n_feat]
        ax.scatter(ml_data['Accuracy'], ml_data['AUC'], alpha=0.6, s=50, 
                  label=f'ML ({n_feat} features)' if n_feat == feature_counts[0] else '', color='#1f77b4')
        ax.scatter(dl_data['Accuracy'], dl_data['AUC'], alpha=0.6, s=50, marker='s',
                  label=f'DL ({n_feat} features)' if n_feat == feature_counts[0] else '', color='#ff7f0e')
    
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('Accuracy vs AUC Score', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/accuracy_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Comparison plots created successfully!")

def main():
    """Main function"""
    print("Loading results from all experiments...")
    results_df = load_all_results()
    
    if results_df is None or len(results_df) == 0:
        print("No results found!")
        return
    
    print(f"Loaded results from {len(results_df)} model evaluations")
    
    # Create comparison plots
    create_comparison_plots(results_df)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("FEATURE COUNT EXPERIMENT SUMMARY")
    print("="*80)
    
    for n_features in [50, 100, 150, 200, 217]:
        subset = results_df[results_df['N_Features'] == n_features]
        ml_subset = subset[subset['Model_Type'] == 'Traditional ML']
        dl_subset = subset[subset['Model_Type'] == 'Deep Learning']
        
        print(f"\n{n_features} Features:")
        print(f"  Traditional ML - Best AUC: {ml_subset['AUC'].max():.4f}, Avg AUC: {ml_subset['AUC'].mean():.4f}")
        print(f"                     Best Accuracy: {ml_subset['Accuracy'].max():.4f}, Avg Accuracy: {ml_subset['Accuracy'].mean():.4f}")
        print(f"  Deep Learning - Best AUC: {dl_subset['AUC'].max():.4f}, Avg AUC: {dl_subset['AUC'].mean():.4f}")
        print(f"                     Best Accuracy: {dl_subset['Accuracy'].max():.4f}, Avg Accuracy: {dl_subset['Accuracy'].mean():.4f}")
    
    print("="*80)

if __name__ == "__main__":
    main()

