import pandas as pd
import numpy as np


mm_graph_selector_scenario = ['Metrics', 'Features', 'Histogram','Scatter']
mm_graph_selected_scenario = mm_graph_selector_scenario[0]

mm_algorithm_selector = ['Baseline', 'ML']
mm_algorithm_selected = 'ML'

mm_pie_color_dict_2 = {"piecolorway":["#00D08A","#FE913C"]}
mm_pie_color_dict_4 = {"piecolorway":["#00D08A","#81F1A0","#F3C178","#FE913C"]}

mm_margin_features = {'margin': {'l': 150}}

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



mm_model_manager_md = """
# **Model**{: .color-primary} Manager

<|layout|columns=3 2 2 2|
<|{mm_graph_selected_scenario}|toggle|lov={mm_graph_selector_scenario}|>


<|{mm_algorithm_selected}|selector|lov={mm_algorithm_selector}|dropdown|label=Algorithm|>

<|show roc|button|on_action={lambda s: s.assign("dr_show_roc", True)}|>

<br/> **Number of predictions:** <|{number_of_predictions}|>
|>

-----------------------------------------------------------------





<Metrics|part|render={mm_graph_selected_scenario == 'Metrics'}|
### Metrics

<|layout|columns=1 1 1|columns[mobile]=1|
<accuracy|
<|{accuracy}|indicator|value={accuracy}|min=0|max=1|>

**Model accuracy**
{: .text-center}

<|{pie_plotly}|chart|title=Accuracy of predictions model|values=values|labels=labels|type=pie|layout={mm_pie_color_dict_2}|>
|accuracy>

<score_auc|
<|{score_auc}|indicator|value={score_auc}|min=0|max=1|>

**Model AUC**
{: .text-center}

<|{pie_confusion_matrix}|chart|title=Confusion Matrix|values=values|labels=labels|type=pie|layout={mm_pie_color_dict_4}|>
|score_auc>

<f1_score|
<|{f1_score}|indicator|value={f1_score}|min=0|max=1|>

**Model F1-score**
{: .text-center}

<|{distrib_class}|chart|title=Distribution between Exited and Stayed|values=values|labels=labels|type=pie|layout={mm_pie_color_dict_2}|>
|f1_score>
|>
|Metrics>





<Features|part|render={mm_graph_selected_scenario == 'Features'}|
### Features
<|{features_table}|chart|type=bar|y=Features|x=Importance|orientation=h|layout={mm_margin_features}|>
|Features>






<Histogram|part|render={mm_graph_selected_scenario == 'Histogram'}|
### Histogram
<|{x_selected}|selector|lov={select_x}|dropdown|label=Select x|>

<|{histo_full_pred}|chart|type=histogram|properties={properties_histo_full}|rebuild|y=PREDICTION|label=PREDICTION|color[1]=red|color[2]=green|name[1]=Good Predictions|name[2]=Bad Predictions|height=600px|>
|Histogram>






<Scatter|part|render={mm_graph_selected_scenario == 'Scatter'}|
### Scatter
<|layout|columns=1 2|
<|{x_selected}|selector|lov={select_x}|dropdown|label=Select x|>

<|{y_selected}|selector|lov={select_y}|dropdown=True|label=Select y|>
|>

<|{scatter_dataset_pred}|chart|properties={properties_scatter_dataset}|rebuild|color[1]=red|color[2]=green|name[1]=Bad prediction|name[2]=Good prediction|mode=markers|type=scatter|height=600px|>
|Scatter>
"""
