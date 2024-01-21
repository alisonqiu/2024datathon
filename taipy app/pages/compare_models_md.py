import numpy as np

from sklearn.metrics import f1_score

import pandas as pd
import numpy as np

cm_height_histo = "100%"
cm_dict_barmode = {"barmode": "stack","margin":{"t":30}}
cm_options_md = "height={cm_height_histo}|width={cm_height_histo}|layout={cm_dict_barmode}"

cm_compare_models_md = """
# Model comparison

----

<br/>
<br/>
<br/>

<|layout|columns=   1 1 1|columns[mobile]=1|
<|{accuracy_graph}|chart|type=bar|x=Model Type|y[1]=Accuracy Model|y[2]=Accuracy Baseline|title=Accuracy|""" + cm_options_md + """|>

<|{f1_score_graph}|chart|type=bar|x=Model Type|y[1]=F1 Score Model|y[2]=F1 Score Baseline|title=F1 Score|""" + cm_options_md + """|>

<|{score_auc_graph}|chart|type=bar|x=Model Type|y[1]=AUC Score Model|y[2]=AUC Score Baseline|title=AUC Score|""" + cm_options_md + """|>

|>
"""

def compare_charts(accuracies, f1_scores, scores_auc, names):
    """This funcion creates the pandas Dataframes (charts) used in the model comparison page

    Args:
        accuracies (list): list of accuracies
        f1_scores (list): list of f1 scores
        scores_auc (list): list of auc scores
        names (list): list of scenario names

    Returns:
        pd.DataFrame: the resulting three pd.DataFrame
    """
    accuracy_graph = pd.DataFrame(create_metric_dict(accuracies, "Accuracy", names))
    f1_score_graph = pd.DataFrame(create_metric_dict(f1_scores, "F1 Score", names))
    score_auc_graph = pd.DataFrame(create_metric_dict(scores_auc, "AUC Score", names))

    return accuracy_graph, f1_score_graph, score_auc_graph

def compare_models_baseline(scenario,model_types):
    """This function creates the objects for the model comparison

    Args:
        scenario (scenario): the selected scenario
        model_types (str): the name of the selected model type

    Returns:
        pd.DataFrame: the resulting three pd.DataFrame
    """
    accuracies = []
    f1_scores = []
    scores_auc = []
    names = []
    for model_type in model_types:
        (_,accuracy,f1_score,score_auc,_,_,_,_,_,_) = c_update_metrics(scenario, model_type)
        
        accuracies.append(accuracy)
        f1_scores.append(f1_score)
        scores_auc.append(score_auc)
        names.append('Model' if model_type != "baseline" else "Baseline") 
        
    accuracy_graph,f1_score_graph, score_auc_graph = compare_charts(accuracies, f1_scores, scores_auc, names)
    return accuracy_graph, f1_score_graph, score_auc_graph
    # Decision Tree Regression
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_predictions = dt_model.predict(X_test)
    dt_rmse = sqrt(mean_squared_error(y_test, dt_predictions))
    print(f"Decision Tree Regression RMSE: {dt_rmse}")

    # Random Forest Regression
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_rmse = sqrt(mean_squared_error(y_test, rf_predictions))
    print(f"Random Forest Regression RMSE: {rf_rmse}")

    # XGBoost Regression
    xgb_model = xgb.XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_predictions = xgb_model.predict(X_test)
    xgb_rmse = sqrt(mean_squared_error(y_test, xgb_predictions))
    print(f"XGBoost Regression RMSE: {xgb_rmse}")

    

def create_metric_dict(metric, metric_name, names):
    """This function creates a dictionary of metrics for multiple models that will be used in a Dataframe shown on the Gui

    Args:
        metric (list): the value of the metric
        metric_name (str): the name of the metric
        names (list): list of scenario names

    Returns:
        dict: dicitonary used for a pandas Dataframe
    """
    metric_dict = {}
    initial_list = [0]*len(names)
    
    metric_dict["Model Type"] = names
    for i in range(len(names)):
        current_list = initial_list.copy()
        
        current_list[i] = metric[i]
        metric_dict[metric_name +" "+ names[i].capitalize()] = current_list
        
    return metric_dict

def c_update_metrics(scenario, model_type):
    """This function updates the metrics of a scenario using a model

    Args:
        scenario (scenario): the selected scenario
        model_type (str): the name of the selected model_type

    Returns:
        obj: a number of values, lists that represent the metrics
    """
    metrics = scenario.data_nodes[f'metrics_{model_type}'].read()

    number_of_predictions = metrics['number_of_predictions']
    number_of_good_predictions = metrics['number_of_good_predictions']
    number_of_false_predictions = metrics['number_of_false_predictions']

    accuracy = np.around(metrics['accuracy'], decimals=2)
    f1_score = np.around(metrics['f1_score'], decimals=2)
    score_auc = np.around(scenario.data_nodes[f'score_auc_{model_type}'].read(), decimals=2)
    
    dict_ftpn = metrics['dict_ftpn']
    
    fp_ = dict_ftpn['fp']
    tp_ = dict_ftpn['tp']
    fn_ = dict_ftpn['fn']
    tn_ = dict_ftpn['tn']
    return number_of_predictions, accuracy, f1_score, score_auc, number_of_good_predictions, number_of_false_predictions, fp_, tp_, fn_, tn_

