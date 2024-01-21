from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from math import sqrt

import datetime as dt

import pandas as pd
import numpy as np

##############################################################################################################################
# Function used in the tasks
##############################################################################################################################

def preprocess_dataset(initial_dataset: pd.DataFrame, date: dt.datetime="None"):
    """This function preprocess the dataset to be used in the model

    Args:
        initial_dataset (pd.DataFrame): the raw format when we first read the data

    Returns:
        pd.DataFrame: the preprocessed dataset for classification
    """
    print("\n     Preprocessing the dataset...")
    
    #We filter the dataframe on the date
    if date != "None":
        initial_dataset['Date'] = pd.to_datetime(initial_dataset['Date'])
        processed_dataset = initial_dataset[initial_dataset['Date'] <= date]
        print(len(processed_dataset))
    else:
        processed_dataset = initial_dataset
        
    processed_dataset = processed_dataset[['CreditScore','Geography','Gender','Age','Tenure','Balance','NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary','OilPeakRate']]
    
    
    processed_dataset = pd.get_dummies(processed_dataset)

    if 'Gender_Female' in processed_dataset.columns:
        processed_dataset.drop('Gender_Female',axis=1,inplace=True)
        
    processed_dataset = processed_dataset.apply(pd.to_numeric)
    
    columns_to_select = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                         'IsActiveMember', 'EstimatedSalary',  'Geography_France', 'Geography_Germany',
                         'Geography_Spain',  'Gender_Male','OilPeakRate']
    
    processed_dataset = processed_dataset[[col for col in columns_to_select if col in processed_dataset.columns]]

    print("     Preprocessing done!\n")
    return processed_dataset


def create_train_test_data(preprocessed_dataset: pd.DataFrame):
    """This function will create the train data by segmenting the dataset

    Args:
        preprocessed_dataset (pd.DataFrame): the preprocessed dataset

    Returns:
        pd.DataFrame: the training dataset
    """
    print("\n     Creating the training and testing dataset...")
    
    X = preprocessed_dataset.drop('OilPeakRate', axis=1)
    y = preprocessed_dataset['OilPeakRate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    train_data = pd.concat([X_train,y_train],axis=1)
    test_data = pd.concat([X_test,y_test],axis=1)


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
        
    print("     Creating done!")
    return X_train, X_test, y_train, y_test, train_data, test_data


def train_model_baseline(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    """Function to train the Logistic Regression model

    Args:
        train_dataset (pd.DataFrame): the training dataset

    Returns:
        model (LogisticRegression): the fitted model
    """
    print("     Training the model...\n")

    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train, y_train)
    linear_reg_predictions = linear_reg_model.predict(X_test)
    linear_reg_rmse = sqrt(mean_squared_error(y_test, linear_reg_predictions))
    return linear_reg_model, linear_reg_rmse, linear_reg_predictions

def train_model_ml(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame):
    #TODO: change algo
    """Function to train the Logistic Regression model

    Args:
        train_dataset (pd.DataFrame): the training dataset

    Returns:
        model (RandomForest): the fitted model
    """
    print("     Training the model...\n")
    linear_reg_model = LinearRegression()
    linear_reg_model.fit(X_train, y_train)
    linear_reg_predictions = linear_reg_model.predict(X_test)
    linear_reg_rmse = sqrt(mean_squared_error(y_test, linear_reg_predictions))
    return linear_reg_model, linear_reg_rmse, linear_reg_predictions

def forecast(test_dataset: pd.DataFrame, trained_model: RandomForestClassifier):
    """Function to forecast the test dataset

    Args:
        test_dataset (pd.DataFrame): the test dataset
        trained_model (LogisticRegression): the fitted model

    Returns:
        forecast (pd.DataFrame): the forecasted dataset
    """
    print("     Forecasting the test dataset...")
    X,y = test_dataset.iloc[:,:-1],test_dataset.iloc[:,-1]
    #predictions = trained_model.predict(X)
    predictions = trained_model.predict_proba(X)[:, 1]
    print("     Forecasting done!")
    return predictions


def roc_from_scratch(probabilities, test_dataset, partitions=100):
    print("     Calculation of the ROC curve...")
    y_test = test_dataset.iloc[:,-1]
    
    roc = np.array([])
    for i in range(partitions + 1):
        threshold_vector = np.greater_equal(probabilities, i / partitions).astype(int)
        tpr, fpr = true_false_positive(threshold_vector, y_test)
        roc = np.append(roc, [fpr, tpr])
    
    roc_np = roc.reshape(-1, 2)
    roc_data = pd.DataFrame({"False positive rate": roc_np[:, 0], "True positive rate": roc_np[:, 1]})
    print("     Calculation done")
    print("     Scoring...")

    score_auc = roc_auc_score(y_test, probabilities)
    print("     Scoring done\n")

    return roc_data, score_auc


def true_false_positive(threshold_vector:np.array, y_test:np.array):
    """Function to calculate the true positive rate and the false positive rate
    
    Args:
        threshold_vector (np.array): the test dataset
        y_test (np.array): the fitted model

    Returns:
        tpr (pd.DataFrame): the forecasted dataset
        fpr (pd.DataFrame): the forecasted dataset
    """
    
    true_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 1)
    true_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 0)
    false_positive = np.equal(threshold_vector, 1) & np.equal(y_test, 0)
    false_negative = np.equal(threshold_vector, 0) & np.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr

def create_metrics(predictions:np.array, test_dataset:np.array):
    print("     Creating the metrics...")
    threshold = 0.5
    threshold_vector = np.greater_equal(predictions, threshold).astype(int)
    
    y_test = test_dataset.iloc[:,-1]
    
    true_positive = (np.equal(threshold_vector, 1) & np.equal(y_test, 1)).sum()
    true_negative = (np.equal(threshold_vector, 0) & np.equal(y_test, 0)).sum()
    false_positive = (np.equal(threshold_vector, 1) & np.equal(y_test, 0)).sum()
    false_negative = (np.equal(threshold_vector, 0) & np.equal(y_test, 1)).sum()


    f1_score = np.around(2*true_positive/(2*true_positive+false_positive+false_negative), decimals=2)
    accuracy = np.around((true_positive+true_negative)/(true_positive+true_negative+false_positive+false_negative), decimals=2)
    dict_ftpn = {"tp": true_positive, "tn": true_negative, "fp": false_positive, "fn": false_negative}
    
    
    number_of_good_predictions = true_positive + true_negative
    number_of_false_predictions = false_positive + false_negative
    
    metrics = {"f1_score": f1_score,
               "accuracy": accuracy,
               "dict_ftpn": dict_ftpn,
               'number_of_predictions': len(predictions),
               'number_of_good_predictions':number_of_good_predictions,
               'number_of_false_predictions':number_of_false_predictions}
    
    return metrics

    
def create_results(forecast_values,test_dataset):
    forecast_series_proba = pd.Series(np.around(forecast_values,decimals=2), index=test_dataset.index, name='Probability')
    forecast_series = pd.Series((forecast_values>0.5).astype(int), index=test_dataset.index, name='Forecast')
    true_series = pd.Series(test_dataset.iloc[:,-1], name="Historical",index=test_dataset.index)
    index_series = pd.Series(range(len(true_series)), index=test_dataset.index, name="Id")
    
    results = pd.concat([index_series, forecast_series_proba, forecast_series, true_series], axis=1)
    return results