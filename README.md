Customer Churn Prediction
A complete machine learning classification project that predicts whether a telecom customer may cancel their service. The project includes synthetic data generation, preprocessing, model comparison, evaluation, saved artifacts, tests, and an interactive Streamlit application.

Features
Generates a reproducible synthetic customer dataset
Handles numerical and categorical data with a scikit-learn pipeline
Compares Logistic Regression and Random Forest
Evaluates accuracy, precision, recall, F1 score, and ROC-AUC
Saves the best model and evaluation results
Includes a confusion matrix and feature-importance chart
Provides an interactive Streamlit prediction app
Includes pytest and a GitHub Actions workflow
Educational portfolio project only. The dataset is synthetic, and predictions should not be used for real customer decisions.

Project structure
customer-churn-prediction/
├── app.py
├── train.py
├── requirements.txt
├── data/
├── models/
├── outputs/
├── src/
│   ├── __init__.py
│   └── data.py
├── tests/
│   └── test_project.py
└── .github/workflows/tests.yml
Run in VS Code on Windows
Open this folder in VS Code and select Terminal > New Terminal.

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train.py
python -m streamlit run app.py
If PowerShell blocks activation:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
Run the test
python -m pytest -q
Upload to GitHub
Create an empty GitHub repository named customer-churn-prediction, then run:

git init
git add .
git commit -m "Add complete customer churn ML project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-prediction.git
git push -u origin main
Author
Abdul Rehman
