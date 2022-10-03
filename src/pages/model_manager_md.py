import pandas as pd
import numpy as np


mm_select_x_ = ['CREDITSCORE', 'AGE', 'TENURE', 'BALANCE', 'NUMOFPRODUCTS', 'HASCRCARD', 'ISACTIVEMEMBER', 'ESTIMATEDSALARY', 'GEOGRAPHY_FRANCE', 'GEOGRAPHY_GERMANY', 'GEOGRAPHY_SPAIN', 'GENDER_MALE']

mm_graph_selector_scenario = ['Metrics', 'Features', 'Histogram','Scatter']
mm_graph_selected_scenario = mm_graph_selector_scenario[0]

mm_algorithm_selector = ['Baseline','ML']
mm_algorithm_selected = mm_algorithm_selector[0]

mm_pie_color_dict_2 ={"piecolorway":["#00D08A","#FE913C"]}
mm_pie_color_dict_4 = {"piecolorway":["#00D08A","#81F1A0","#F3C178","#FE913C"]}

mm_height_histo = 530


mm_margin_features = {'margin': {'l': 150, 'r': 50, 'b': 50, 't': 20}}

def creation_scatter_dataset_pred(test_dataset:pd.DataFrame, forecast_series:pd.Series):
    """This function creates the dataset for the scatter plot for the predictions.  For every column (except EXITED) will have a positive and negative version.
    EXITED is here a binary indicating if the prediction is good or bad.
    The positive column will have NaN when the Exited is zero and the negative column will have NaN when the Exited is one. 

    Args:
        test_dataset (pd.DataFrame): the test dataset
        forecast_series (pd.DataFrame): the forecast dataset

    Returns:
        pd.DataFrame: the Dataframe used to display the Histogram
    """
    
    scatter_dataset = test_dataset.copy()
    scatter_dataset['EXITED'] =  (scatter_dataset['EXITED']!=forecast_series.to_numpy()).astype(int)

    for column in scatter_dataset.columns:
        if column != 'EXITED' :
            column_neg = str(column)+'_neg'
            column_pos = str(column)+'_pos'
            
            scatter_dataset[column_neg] = scatter_dataset[column]
            scatter_dataset[column_pos] = scatter_dataset[column]
            
            scatter_dataset.loc[(scatter_dataset['EXITED'] == 1),column_neg] = np.NaN
            scatter_dataset.loc[(scatter_dataset['EXITED'] == 0),column_pos] = np.NaN
    
    return scatter_dataset


def creation_of_dialog_scatter_pred(column, state=None):
    """This code generates the Markdown used for the scatter plot for the predictions. It is going to be used to change 
    the partial (mini-page that can be reloaded). Selectors are created and the x and y of the graph is determined by changing it
    here. The dictionary of properties is also changed depending on the column used.
    """
    if column == 'AGE' or column == 'CREDITSCORE' and state is not None:
        state.dv_dict_overlay = {'barmode':'overlay',"margin":{"t":20}}
    elif state is not None:
        state.dv_dict_overlay = {"margin":{"t":20}}
        
    md = """
<|layout|columns= 1 1|columns[mobile]=1|
<|
Select **x** \n \n <|{x_selected}|selector|lov={select_x}|dropdown|>
|>

<|
Select **y** \n \n <|{y_selected}|selector|lov={select_y}|dropdown|>
|>
|>

<|{scatter_dataset_pred}|chart|x="""+column+"""|y[1]={y_selected+'_pos'}|y[2]={y_selected+'_neg'}|color[1]=red|color[2]=green|name[1]=Bad prediction|name[2]=Good prediction|height={mm_height_histo}|width={dv_width_histo}|mode=markers|type=scatter|layout={dv_dict_overlay}|>

"""
    return md


def creation_histo_full_pred(test_dataset:pd.DataFrame,forecast_series:pd.Series):
    """This function creates the dataset for the histogram plot for the predictions.  For every column (except PREDICTION) will have a positive and negative version.
    PREDICTION is a binary indicating if the prediction is good or bad.
    The positive column will have NaN when the PREDICTION is zero and the negative column will have NaN when the PREDICTION is one. 

    Args:
        test_dataset (pd.DataFrame): the test dataset
        forecast_series (pd.DataFrame): the forecast dataset

    Returns:
        pd.DataFrame: the Dataframe used to display the Histogram
    """
    histo_full = test_dataset.copy()
    histo_full['EXITED'] =  (histo_full['EXITED']!=forecast_series.to_numpy()).astype(int)
    histo_full.columns = histo_full.columns.str.replace('EXITED', 'PREDICTION')
    
    for column in histo_full.columns:
        column_neg = str(column)+'_neg'
        histo_full[column_neg] = histo_full[column]
        histo_full.loc[(histo_full['PREDICTION'] == 1),column_neg] = np.NaN
        histo_full.loc[(histo_full['PREDICTION'] == 0),column] = np.NaN
        
    return histo_full


metrics_md = """
<br/>
<|layout|columns=1 1 1|columns[mobile]=1|

<|
<|{accuracy}|indicator|value={accuracy}|min=0|max=1|>
<center>
**Model accuracy**
</center>
<|{pie_plotly}|chart|x=values|label=labels|title=Accuracy of predictions model|height={height_plotly}|width=100%|type=pie|layout={mm_pie_color_dict_2}|>
|>

<|
<|{score_auc}|indicator|value={score_auc}|min=0|max=1|>
<center>
**Model AUC**
</center>
<|{pie_confusion_matrix}|chart|x=values|label=labels|title=Confusion Matrix|height={height_plotly}|width=100%|type=pie|layout={mm_pie_color_dict_4}|>
|>

<|
<|{f1_score}|indicator|value={f1_score}|min=0|max=1|>
<center>
**Model F1-score**
</center>
<|{distrib_class}|chart|x=values|label=labels|title=Distribution between Exited and Stayed|height={height_plotly}|width=100%|type=pie|layout={mm_pie_color_dict_2}|>
|>

|>
"""


features_md = """
<|{features_table}|chart|type=bar|y=Features|x=Importance|orientation=h|layout={mm_margin_features}|>
"""

def creation_of_dialog_histogram_pred(column, state=None):
    """This code generates the Markdown used for the scatter plot. It is going to be used to change 
    the partial (mini-page that can be reloaded). The selector is created and the x of the graph is determined by changing it
    here. The dictionary of properties is also changed depending on the column used.
    """
    if column == 'AGE' or column == 'CREDITSCORE' and state is not None:
        state.dv_dict_overlay = {'barmode':'overlay',"margin":{"t":20}}
    elif state is not None:
        state.dv_dict_overlay = {"margin":{"t":20}}
        
    md = """
<|
Select **x** \n \n <|{x_selected}|selector|lov={select_x}|dropdown=True|>
|>

<|{histo_full_pred[['"""+column+"""','"""+column+"""_neg','PREDICTION']]}|chart|type=histogram|x[1]="""+column+"""|x[2]="""+column+"""_neg|y=PREDICTION|label=PREDICTION|color[1]=red|color[2]=green|name[1]=Bad prediction|name[2]=Good prediction|height={mm_height_histo}|width={dv_width_histo}|layout={dv_dict_overlay}|class_name=histogram|>

"""
    return md


mm_model_manager_md = """
# Model Manager

<|layout|columns=1 1 1 1|columns[mobile]=1|
Algorithm
<|{mm_algorithm_selected}|selector|lov={mm_algorithm_selector}|dropdown=True|>

Type of graph
<|{mm_graph_selected_scenario}|selector|lov={mm_graph_selector_scenario}|dropdown=True|>

<br/>
<br/>
<center>
<|show roc|button|on_action=show_roc_fct|>
</center>

<br/>
<br/>
<center>
**Number of predictions: ** *<|{number_of_predictions}|>*
</center>
|>

<|part|render={mm_graph_selected_scenario == 'Metrics'}|
"""+metrics_md+"""
|>

<|part|render={mm_graph_selected_scenario == 'Features'}|
"""+features_md+"""
|>

<|part|render={mm_graph_selected_scenario == 'Scatter'}|partial={partial_scatter_pred}|>

<|part|render={mm_graph_selected_scenario == 'Histogram'}|partial={partial_histo_pred}|>

"""
