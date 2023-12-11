Credit Card Fraud Detection Using Different Models.
Introduction
This project focuses on detecting credit card fraud using a Decision Tree classifier, ANN and Logistic regression. The aim is to accurately identify fraudulent transactions from a given dataset and with different models.

Data Source
The dataset used in this project is a credit card transaction dataset. It contains features related to transaction details and a target variable indicating whether a transaction is fraudulent or not.

Dependencies
Python 3.x
Pandas
NumPy
Scikit-Learn
Imbalanced-Learn
Matplotlib
Seaborn

Installation
To run this project, Python 3.x is required along with the above-listed libraries. These can be installed via pip:
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn

Usage
To execute the project, run the Jupyter notebook credit_card_project_analysis.ipynb. The notebook includes data loading, preprocessing, model training, and evaluation steps.

Methodology
Data Preparation: The data is first loaded and then split into features and target. The features are then scaled using StandardScaler.
Handling Imbalance: The SMOTE technique is used to balance the dataset.
Model Training: A Decision Tree classifier is trained on the balanced dataset.
Evaluation: The model is evaluated using metrics such as classification report, confusion matrix, ROC curve, and Precision-Recall curve.
Results
The results section includes a summary of the model's performance using different methods and model training. Include key metrics like accuracy, precision, recall, and the AUC score.

Limitations and Future Work
Our primary constraint is the use of a single dataset. In our future endeavors, we plan to expand our analysis to datasets from various countries, employing both the current and different models. This approach aims to enhance our results, particularly focusing on advanced privacy techniques, given the sensitive nature of the data.

Contributors:
Ekta Pandey
Siddhant Jeevanlal Gupta.
Nitin Patel.
Shashwat Jain.
Rishabh Saxena.
