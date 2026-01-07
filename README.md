# Allergen Prediction: ML/DL Evaluation

Machine learning and deep learning framework for predicting chemical compound allergenicity using molecular descriptors.

## Quick Start

```bash
# Setup
python3.12 -m venv env
source env/bin/activate
sed '/^audioop-lts/d' requirements.txt | pip install -r /dev/stdin

# Run experiments
python convert_sdf_to_descriptors.py                    # Step 1: Extract descriptors
python run_feature_count_experiments.py                 # Step 2: Run experiments
python run_experiments_with_hyperparameter_search.py   # Step 3: Optimize hyperparameters (optional)
python visualize_feature_count_results.py              # Step 4: Visualize results
```

## Dataset

- **Training**: 1,158 compounds (315 positive, 843 negative)
- **Test**: 295 compounds (83 positive, 212 negative)
- **Features**: 217 molecular descriptors (RDKit)

## Models

**Traditional ML (9)**: SVM, Random Forest, Gradient Boosting, KNN, Logistic Regression, Neural Network, AdaBoost, Naive Bayes, Decision Tree

**Deep Learning (4)**: MLP, CNN, LSTM, Transformer (with hyperparameter optimization)

## Experiments

Five feature count experiments: 50, 100, 150, 200, 217 features. Results saved in `results/experiment*/`.

## Outputs

- Performance summaries (CSV)
- Comparison plots (PNG)
- Hyperparameter search results (JSON, if Step 3 completed)

## Requirements

See `requirements.txt`. Key: pandas, numpy, scikit-learn, torch, matplotlib, seaborn, RDKit.
