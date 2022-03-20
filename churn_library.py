# library doc string
'''
Churn project to create a clean version of model prediction that meets standard coding standards

Author: Ji Park
Date: 3/19/2022
'''

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # plot parameter setting
    plt.figure(figsize=(20, 10))
    # saves the distribution of churn and customer age to the images
    cols_to_plot = ['Churn', 'Customer_Age', 'Marital_Status']
    for col in cols_to_plot:
        if col == 'Marital_Status':
            df.Marital_Status.value_counts('normalize').plot(kind='bar')
            plt.savefig('images/eda/' + col + '.png')
            plt.clf()
        else:
            df[col].hist()
            plt.savefig('images/eda/' + col + '.png')
            plt.clf()
    # saves the correlation plot (bivariate relationship)
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/eda/corr.png')
    plt.clf()

def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    new code is a vectorized approach

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
    output:
            df: pandas dataframe with new columns for
    '''
    mean_agg_calculate_cols = category_lst

    for agg_col in mean_agg_calculate_cols:
        temp_groups = df.groupby(agg_col).mean()['Churn']

        conditions = [df[agg_col] == cond for cond in list(temp_groups.index)]

        values = list(temp_groups.values)
        new_col_name = agg_col + '_Churn'

        df[new_col_name] = np.select(conditions, values)

    return df

def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: optional string response argument

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

    X, y = pd.DataFrame(), df['Churn']

    X[response] = df[response]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    this function first outputs classification report as a dictionary and
    then uses seaborn for plotting
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    rf_test_report = classification_report(
        y_test, y_test_preds_rf, output_dict=True)
    rf_train_report = classification_report(
        y_train, y_train_preds_rf, output_dict=True)

    lr_test_report = classification_report(
        y_test, y_test_preds_lr, output_dict=True)
    lr_train_report = classification_report(
        y_train, y_train_preds_lr, output_dict=True)

    report_list = [rf_test_report, rf_train_report, lr_test_report, lr_train_report]
    report_names = ["rf_test_report", "rf_train_report", "lr_test_report", "lr_train_report"]
    for report in report_list:
        sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True)
        filename = "./images/results/" + report_names[report_list.index(report)] + '.png'
        plt.savefig(filename)
        plt.clf()

def feature_importance_plot(model, X_data):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar",show=False)
    plt.savefig("./images/results/top_drivers.png")
    plt.clf()

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig("./images/results/feature_importance.png")
    plt.clf()

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              cv_rfc: RandomForest model
              lrc: LogisticRegression model
    '''
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression()

    param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt'],
    'max_depth' : [4,5,100],
    'criterion' :['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    # save ROC curves
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.savefig("./images/results/lrc_plot.png")
    plt.clf()

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig("./images/results/lrc_rfc_plot.png")
    plt.clf()

    # saves the models
    rfc_filepath = './models/rfc_model.pkl'
    joblib.dump(cv_rfc.best_estimator_, rfc_filepath)
    lrc_filepath = './models/logistic_model.pkl'
    joblib.dump(lrc, lrc_filepath)

    return cv_rfc, lrc
    