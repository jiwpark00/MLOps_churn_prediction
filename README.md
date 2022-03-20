# MLOps_churn_prediction
MLOps Project 1 for Udacity's Machine Learning DevOps Program

## Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project applies modularized and refactored approach to transfrom a prototype level code into a production-level code. Core functions used in the churn_notebook (Jupyter) are defined and called from churn_library.py file with relevant functions being tested in churn_script_logging_and_testing file.
This project also utilized pylint and autopep8 to streamline syntax and code style to meet the PEP8 standards.

## Running Files
First, you have to ensure your scikit-learn version is up-to-date at least 0.23.2 - otherwise shap library import will fail.
Second, follow through churn_notebook.ipynb and call the functions in churn_library that is imported there. If there is any error, churn_script_logging_and_tests file includes potential source of error in the data or processing step.

## Files in the Repo
Each bullet refers to a folder or a file at the root level.
* data -- bank_data.csv (source of data)
* images <br />
-- eda folder: <br />
--- Churn.png (Churn column EDA) <br />
--- corr.png (correlation of bivariate column relationship EDA) <br />
--- Customer_Age.png (Customer_Age column EDA) <br />
--- Marital_Status.png (Marital_Status column EDA) <br />
-- results folder: <br />
--- feature_importance.png (random forest model feature importance plot) <br />
--- top_drivers.png (random forest model SHAP-based top drivers plot) <br />
--- lrc_plot.png (logistic regression model ROC curve) <br />
--- lrc_rfc_plot.png (logistic regression and random forest model ROC curve) <br />
--- lr_test_report (classification report for logistic regression model on test partition) <br />
--- lr_train_report (classification report for logistic regression model on train partition) <br />
--- rf_test_report (classification report for random forest model on test partition) <br />
--- rf_train_report (classification report for random forest model on train partition) <br />
* log -- churn_library.png (logs from logging and testing file)
* models <br />
--- logistic_model.pkl (logistic regression model saved in pickle format) <br />
--- rfc_model.pkl (random forest model saved in pickle format) <br />
* churn_library.py -- houses main core functions for data science work in this project
* churn_notebook.ipynb -- calls the functions in churn_library - recommended to use this notebook and keep churn_library and churn_script_logging_and_tests at the same level 
* churn_script_logging_and_tests.py -- houses tests for the main core functions in churn_library.py
* README.md -- this file; main documentation for how to use this repo

## Running testing file
ipython churn_script_logging_and_tests_solution.py or python churn_script_logging_and_tests_solution.py depending on your configuration