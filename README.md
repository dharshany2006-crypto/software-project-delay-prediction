# Software Project Delay Prediction Using Machine Learning

## Project Description

This project uses Machine Learning algorithms to predict whether a software development project is likely to be delayed or completed on time.

The system analyses different project factors such as team size, project duration, budget, requirement changes, team experience, project complexity and bugs.

It predicts the probability of project delay and classifies the project risk into three categories:

- LOW RISK
- MEDIUM RISK
- HIGH RISK

## Machine Learning Algorithms

This project uses three classification algorithms:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier

The algorithms are compared to identify the best-performing model.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Joblib

## Features

- Dataset generation
- Data exploration and analysis
- Machine Learning model training
- Model performance comparison
- Accuracy, Precision, Recall and F1-score evaluation
- Confusion matrix visualization
- ROC curve visualization
- Feature importance analysis
- New project delay prediction
- Risk classification and recommendations

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Run the following command:

```bash
python project_delay_prediction.py
```

## Output

The program generates:

- Project dataset
- Model comparison results
- Correlation graph
- Confusion matrix
- ROC curve
- Feature importance graph
- Trained Machine Learning model

All generated output files are stored in the `outputs` folder.

## Dataset

This version uses a synthetically generated dataset containing 2,000 project records.

The synthetic dataset is intended for educational and demonstration purposes. Real-world data is required for practical prediction validation.

## Author

Student Machine Learning Project

## Project Title

Software Project Delay Prediction Using Machine Learning
