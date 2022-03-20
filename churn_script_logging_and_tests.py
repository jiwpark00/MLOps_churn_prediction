'''
Testing and logging file to check for normal runs for churn_library

Author: Ji Park
Created: 3/20/2022
'''

import os
import logging
import churn_library as cls
import pytest
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, df):
    '''
    test perform eda function
    '''
    # this is necessary as it's not in the churn_library but in Jupyter
    # notebook
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    try:
        # check for the presence of the column
        cols_to_test = ["Churn", "Customer_Age", "Marital_Status"]
        assert cols_to_test[0] in df.columns
        assert cols_to_test[1] in df.columns
        assert cols_to_test[2] in df.columns
        perform_eda(df)
        logging.info("SUCCESS on EDA")
    except AssertionError as err:
        logging.error(
            "EDA code did not run work on the input dataframe col missing")


def test_encoder_helper(encoder_helper, df):
    '''
    test encoder helper
    we check if transform successfully worked by checking the final # of
    columns after encoding
    '''
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    try:
        encoder_helper(df, category_lst)
        assert df.shape[1] == 28
        logging.info("SUCCESS: encoding is successful for mean agg churns")
    except AssertionError:
        logging.error(str(df.shape[1]))  # of columns
        logging.error("Error: encoding failed")


def test_perform_feature_engineering(perform_feature_engineering, df):
    '''
    test perform_feature_engineering
    '''
    keeps_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    try:
        assert df[keeps_cols].shape[1] == len(
            keeps_cols)  # checks all columns present
        perform_feature_engineering(df, keeps_cols)
        logging.info("SUCCESS: Train and Test Split complete")
    except AssertionError:
        logging.error("Error: Check the column format/name")


def test_train_models(train_models, X_train, X_test, y_train, y_test):
    '''
    test train_models
    '''
    try:
        # check the shape of y_train and y_test - e.g., only 1 column
        assert len(list(y_train.shape)) == 1
        assert len(list(y_test.shape)) == 1
        train_models(X_train, X_test, y_train, y_test)
        logging.info("SUCCESS: Model is trained")
    except AssertionError:
        logging.error("Error: Model building didn't work")


if __name__ == "__main__":
    test_import(cls.import_data)
    INPUT = cls.import_data("./data/bank_data.csv")
    test_eda(cls.perform_eda, INPUT)
    INPUT['Churn'] = INPUT['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    test_encoder_helper(cls.encoder_helper, INPUT)
    category_list = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    INPUT = cls.encoder_helper(INPUT, category_list)
    test_perform_feature_engineering(cls.perform_feature_engineering, INPUT)
    KEEPS_COLS = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']
    X_TRAIN, X_TEST, y_TRAIN, y_TEST = cls.perform_feature_engineering(
        INPUT, KEEPS_COLS)
    test_train_models(cls.train_models, X_TRAIN, X_TEST, y_TRAIN, y_TEST)
