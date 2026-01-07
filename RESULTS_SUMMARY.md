# Experimental Results Summary

## Overview
The hyperparameter search and optimized deep learning evaluation completed for all 5 experiments (50, 100, 150, 200, 217 features). However, **critical failures were detected** in 3 model configurations.

## Successful Results

### Best Performing Models (by AUC)
1. **Experiment 4 (200 features) - Transformer**: AUC = 0.931, Accuracy = 0.881
2. **Experiment 1 (50 features) - Basic MLP**: AUC = 0.931, Accuracy = 0.847
3. **Experiment 4 (200 features) - CNN**: AUC = 0.908, Accuracy = 0.885
4. **Experiment 1 (50 features) - CNN**: AUC = 0.901, Accuracy = 0.871

### Model Performance by Architecture (Average AUC across all experiments)
- **Basic MLP**: ~0.916 (consistent across all feature counts)
- **Transformer**: ~0.891 (best at 200 features: 0.931)
- **CNN**: ~0.840 (highly variable, see failures below)
- **LSTM**: ~0.780 (highly variable, see failures below)

## Critical Failures Detected

### Failed Model Configurations
Three models completely failed during optimized evaluation, producing baseline (majority class) predictions:

1. **Experiment 2 (100 features) - CNN**
   - AUC: 0.5 (random)
   - Accuracy: 0.7186 (baseline - predicting all negative)
   - Precision: 0.0, Recall: 0.0, F1: 0.0
   - **Note**: Hyperparameter search showed AUC = 0.914 for this model

2. **Experiment 3 (150 features) - LSTM**
   - AUC: 0.508 (near-random)
   - Accuracy: 0.7186 (baseline)
   - Precision: 0.0, Recall: 0.0, F1: 0.0
   - **Note**: Hyperparameter search showed AUC = 0.911 for this model

3. **Experiment 5 (217 features) - CNN**
   - AUC: 0.5 (random)
   - Accuracy: 0.7186 (baseline)
   - Precision: 0.0, Recall: 0.0, F1: 0.0
   - **Note**: Hyperparameter search showed AUC = 0.932 for this model

### Analysis of Failures

**Pattern Identified:**
- All failed models show accuracy = 0.7186, which exactly matches the baseline (213 negative / 297 total test samples)
- This indicates the models are predicting **all samples as negative class**
- The fact that hyperparameter search succeeded but final evaluation failed suggests:
  1. **Bug in model instantiation** during optimized evaluation
  2. **Data preprocessing mismatch** between hyperparameter search and final evaluation
  3. **Model architecture issue** with certain input dimensions (100, 150, 217 features)
  4. **Training failure** (models not learning, producing constant outputs)

**Specific Concerns:**
- CNN failures at 100 and 217 features (but worked at 50, 150, 200)
- LSTM failure at 150 features (but worked at other feature counts)
- The inconsistency suggests the issue is not simply "CNN/LSTM don't work" but rather a specific bug or edge case

## Hyperparameter Search Results

All hyperparameter searches completed successfully. The best hyperparameters were found for all models in all experiments. The search results show:
- MLP: Consistent performance across all feature counts (AUC ~0.93)
- CNN: Strong performance during search (AUC 0.91-0.93) but failures in final evaluation
- LSTM: Good performance during search (AUC 0.90-0.91) but one failure in final evaluation
- Transformer: Good performance (AUC 0.88-0.93)

## Recommendations

1. **Investigate the failed models immediately** - The discrepancy between hyperparameter search results and final evaluation is a critical bug that needs to be fixed.

2. **Check model instantiation code** - Verify that `FlexibleCNN` and `FlexibleLSTM` are being created correctly with the optimized hyperparameters.

3. **Verify data consistency** - Ensure the same preprocessing pipeline is used in both hyperparameter search and final evaluation.

4. **Check for numerical issues** - The models may be producing NaN or constant outputs. Add debugging to check model outputs during training.

5. **Re-run failed experiments** - After fixing the bug, re-run experiments 2, 3, and 5 for CNN and LSTM models.

## Overall Assessment

**Strengths:**
- Hyperparameter search completed successfully for all models
- MLP and Transformer models show consistent, strong performance
- Best overall performance achieved with 200 features (Transformer: AUC 0.931)

**Critical Issues:**
- 3 model configurations completely failed (predicting baseline)
- Discrepancy between hyperparameter search and final evaluation suggests a bug
- Results for CNN and LSTM are unreliable due to failures

**Next Steps:**
1. Debug and fix the model evaluation code
2. Re-run failed experiments
3. Update manuscript with corrected results
4. Consider excluding failed model configurations from final analysis if bug cannot be resolved

