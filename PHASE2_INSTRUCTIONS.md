# Phase 2: Comparative Evaluation Instructions

## Overview

The study has been restructured into two phases:

1. **Phase 1: Hyperparameter Search** (COMPLETED)
   - Comprehensive hyperparameter optimization for all deep learning architectures
   - Results saved in `results/experiment*/hyperparameter_search/`
   - Visualizations can be generated with `visualize_hyperparameter_search_results.py`

2. **Phase 2: Comparative Evaluation** (TO RUN)
   - Re-run evaluation using best architectures from Phase 1
   - Results will be saved in `results/phase2_comparative_evaluation/`
   - Uses fixed evaluation code that addresses bugs found in initial runs

## Running Phase 2

### Prerequisites
- Phase 1 hyperparameter search must be completed
- All experiment directories must exist with hyperparameter search results

### Command
```bash
cd /home/eo/allergen-prediction
source env/bin/activate  # or your virtual environment
python3 phase2_comparative_evaluation.py
```

### What It Does
1. Loads best hyperparameters from Phase 1 for each experiment
2. Creates models with optimized architectures
3. Trains and evaluates on test set
4. Saves results to `results/phase2_comparative_evaluation/experiment*/`
5. Generates overall summary CSV

### Key Fixes in Phase 2
- **Fixed CNN architecture**: Uses dynamic output size calculation to prevent dimension mismatches
- **Fixed label tensors**: Uses FloatTensor instead of LongTensor for binary classification
- **Better error handling**: Catches and reports issues during training
- **Validation checks**: Warns if models produce suspicious results (AUC < 0.5, Precision = 0)

## Visualizing Phase 1 Results

To create figures from Phase 1 hyperparameter search:

```bash
python3 visualize_hyperparameter_search_results.py
```

This creates figures in `results/phase1_hyperparameter_search_figures/`:
- AUC across feature counts
- Accuracy across feature counts
- Heatmaps for AUC and Accuracy
- Best model by feature count
- Comprehensive comparison plots

## Expected Output Structure

```
results/
├── experiment1/
│   └── hyperparameter_search/
│       ├── best_hyperparameters.json
│       └── hyperparameter_search_summary.csv
├── experiment2/
│   └── hyperparameter_search/
│       ├── best_hyperparameters.json
│       └── hyperparameter_search_summary.csv
├── ...
├── phase1_hyperparameter_search_figures/
│   ├── auc_across_features.png
│   ├── accuracy_across_features.png
│   ├── auc_heatmap.png
│   ├── accuracy_heatmap.png
│   ├── best_model_by_features.png
│   └── comprehensive_comparison.png
└── phase2_comparative_evaluation/
    ├── experiment1/
    │   ├── performance_summary.csv
    │   ├── mlp_model.pt
    │   ├── cnn_model.pt
    │   ├── lstm_model.pt
    │   └── transformer_model.pt
    ├── experiment2/
    │   └── ...
    └── overall_summary.csv
```

## Troubleshooting

### If Phase 2 fails for specific models:
- Check the error message - it will indicate which model/experiment failed
- Verify that Phase 1 hyperparameter search completed successfully
- Check that feature-selected descriptor files exist in experiment directories

### If CNN models fail:
- The fixed CNN uses dynamic size calculation, which should prevent dimension errors
- If issues persist, check the input dimension and kernel sizes in best_hyperparameters.json

### If results seem suspicious:
- The script will warn if AUC < 0.5 or Precision = 0
- Compare Phase 2 results with Phase 1 hyperparameter search results
- If they differ significantly, there may be a data preprocessing issue

