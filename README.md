# Allergen Prediction Machine Learning Framework

This repository contains a comprehensive machine learning framework for predicting chemical compound allergenicity using molecular descriptors.

## 📁 Project Structure

```
allergen-prediction/
├── ml_evaluation/                    # Traditional Machine Learning
│   ├── ml_evaluation.py             # Main ML evaluation script
│   └── ml_requirements.txt          # ML dependencies
├── deep_learning_evaluation/         # Deep Learning (PyTorch)
│   ├── evaluation.py                # Deep learning evaluation script
│   └── requirements.txt             # Deep learning dependencies
├── results/                         # All evaluation results
│   ├── ml_evaluation/              # Traditional ML results
│   ├── deep_learning_evaluation/   # Deep learning results
│   └── *.csv                       # Original data files
├── comprehensive_comparison.py      # Combined analysis script
└── README.md                       # This file
```

## 🎯 Dataset Overview

- **Training Set**: 1,177 samples (320 positive, 857 negative)
- **Test Set**: 295 samples
- **Features**: 714 molecular descriptors extracted using PaDEL
- **Task**: Binary classification (allergenic vs non-allergenic)

## 🤖 Models Evaluated

### Traditional Machine Learning (9 models)
- **SVM** - Support Vector Machine
- **Random Forest** - Random Forest Classifier
- **Gradient Boosting** - Gradient Boosting Classifier
- **K-Nearest Neighbors** - KNN Classifier
- **AdaBoost** - AdaBoost Classifier
- **Neural Network** - Multi-layer Perceptron
- **Logistic Regression** - Logistic Regression
- **Decision Tree** - Decision Tree Classifier
- **Naive Bayes** - Gaussian Naive Bayes

### Deep Learning Models (4 PyTorch models)
- **Simple NN** - Simple Neural Network (4 layers)
- **Deep NN** - Deep Neural Network (7 layers)
- **Wide NN** - Wide Neural Network (5 layers, 2048 neurons)
- **Ensemble NN** - Multi-branch Ensemble Network

## 🏆 Key Results

### Best Overall Model
- **Model**: SVM (Traditional ML)
- **AUC Score**: 0.9293 (92.93%)
- **Accuracy**: 85.76%
- **Precision**: 90.20%

### Top 5 Models by AUC Score
1. **SVM** (Traditional ML) - AUC: 0.9293
2. **Random Forest** (Traditional ML) - AUC: 0.9282
3. **Gradient Boosting** (Traditional ML) - AUC: 0.9020
4. **Wide NN** (Deep Learning) - AUC: 0.8937
5. **Deep NN** (Deep Learning) - AUC: 0.8920

### Model Type Performance
- **Traditional ML Average AUC**: 0.8635
- **Deep Learning Average AUC**: 0.8817
- **Best Traditional ML**: SVM (AUC: 0.9293)
- **Best Deep Learning**: Wide NN (AUC: 0.8937)

## 🚀 Usage

### Run Traditional ML Evaluation
```bash
cd ml_evaluation
pip install -r ml_requirements.txt
python ml_evaluation.py
```

### Run Deep Learning Evaluation (PyTorch)
```bash
cd deep_learning_evaluation
pip install -r requirements.txt
python evaluation.py
```

### Run Comprehensive Comparison
```bash
python comprehensive_comparison.py
```

## 📊 Generated Outputs

### Performance Summaries
- `ml_evaluation/model_performance_summary.csv` - Traditional ML results
- `deep_learning_evaluation/performance_summary.csv` - Deep learning results
- `comprehensive_results_summary.csv` - Combined results

### Visualizations
- **Performance comparison charts** for all models
- **ROC curves** and **Precision-Recall curves**
- **Confusion matrices** for each model
- **Feature importance analysis**
- **Training history plots** for deep learning models
- **Model type performance comparison** (box plots and bar charts)
- **Performance scatter plot** (Accuracy vs AUC)

## 🔧 Dependencies

### Traditional ML
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- Standard Python scientific stack

### Deep Learning
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- PyTorch, torchvision

## 📈 Key Insights

1. **SVM performs best** with the highest AUC score of 0.9293
2. **Random Forest is close second** with AUC of 0.9282
3. **Deep learning models show promise** with Wide NN achieving 0.8937 AUC
4. **Traditional ML models are more consistent** across different metrics
5. **Feature importance analysis** reveals key molecular descriptors for allergen prediction

## 🎯 Conclusion

The evaluation demonstrates that both traditional machine learning and deep learning approaches can effectively predict allergenicity from molecular descriptors. The SVM model emerges as the best performer, suggesting that the problem benefits from non-linear decision boundaries and margin maximization. The results provide a solid foundation for building a practical allergen prediction system.

## 📝 Notes

- All results are automatically saved to the `results/` directory
- The framework supports both CPU and GPU training (PyTorch)
- Early stopping and learning rate scheduling are implemented for deep learning models
- Comprehensive evaluation includes multiple metrics and visualizations