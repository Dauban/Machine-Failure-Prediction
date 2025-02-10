# Machine-Failure-Prediction
A machine learning project for predicting machine failures using Logistic Regression, Decision Trees, and Naive Bayes. Includes feature engineering, data preprocessing, model evaluation (precision, recall, accuracy), and hyperparameter tuning with GridSearchCV. Implements feature selection to optimize model performance.

# Machine Failure Prediction

## Overview
This project applies **machine learning models** to predict **machine failures** using **classification algorithms** such as **Logistic Regression, Decision Trees, and Naive Bayes**. It includes **feature engineering, data preprocessing, model evaluation, and hyperparameter tuning** to optimize predictive performance.

## Features
- **Data Preprocessing**:
  - Handles categorical variables through encoding.
  - Normalizes numerical features using **MinMaxScaler**.
  - Splits the dataset into **training and test sets**.
- **Model Training & Evaluation**:
  - Implements **Logistic Regression, Decision Tree, and Naive Bayes classifiers**.
  - Uses **precision, recall, accuracy, and confusion matrices** for performance evaluation.
- **Feature Selection & Optimization**:
  - Identifies the most important features and retrains models without them.
  - Uses **GridSearchCV** for hyperparameter tuning.
- **Visualization**:
  - Displays **confusion matrices** and model performance comparisons.
  - Plots feature importance analysis.

## Technologies Used
- **Python** (NumPy, Pandas, Matplotlib, Seaborn)
- **Scikit-Learn** (Logistic Regression, Decision Tree, Naive Bayes, GridSearchCV)

## Project Structure
```
Machine-Failure-Prediction/
│── data/                     # Dataset (CSV file loaded from Google Drive)
│── model_training.py         # Model training and evaluation
│── feature_engineering.py    # Feature processing and selection
│── model_comparison.py       # Evaluates models and generates confusion matrices
│── README.md                 # Project documentation
```

## Installation & Usage
### Prerequisites
Ensure you have **Python 3.x** and the required libraries installed:
```sh
pip install numpy pandas matplotlib scikit-learn
```

### Running the Project
Run the **main script** to preprocess the data, train models, and evaluate results:
```sh
python model_training.py
```

## Example Output
**Sample Accuracy Metrics:**
```
Logistic Regression:
Accuracy: 99.62%
Precision: 99.12%
Recall: 76.71%

Decision Tree:
Accuracy: 98.55%
Naive Bayes:
Accuracy: 97.77%
```

## Potential Enhancements
- **Integrate Deep Learning (TensorFlow/PyTorch) for improved prediction.**
- **Deploy the model via a REST API (Flask/FastAPI).**
- **Automate feature selection using SHAP or LIME for explainability.**

## Author
**Or Adar**  
GitHub: [github.com/Dauban](https://github.com/Dauban)  

---
This project showcases applied machine learning techniques for industrial failure prediction and optimization.

