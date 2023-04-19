import pandas as pd
import numpy as np


dv_graph_selector = ['Histogram','Scatter']
dv_graph_selected = dv_graph_selector[0]

# Histograms dialog
properties_histo_full = {}
properties_scatter_dataset = {}

def creation_scatter_dataset(test_dataset:pd.DataFrame):
    """This function creates the dataset for the scatter plot.  For every column (except Exited), scatter_dataset will have a positive and negative version.
    The positive column will have NaN when the Exited is zero and the negative column will have NaN when the Exited is one.

    Args:
        test_dataset (pd.DataFrame): the test dataset

    Returns:
        pd.DataFrame: the datafram
    """
    scatter_dataset = test_dataset.copy()

    for column in scatter_dataset.columns:
        if column != 'EXITED' :
            column_neg = str(column)+'_neg'
            column_pos = str(column)+'_pos'
            
            scatter_dataset[column_neg] = scatter_dataset[column]
            scatter_dataset[column_pos] = scatter_dataset[column]
            
            scatter_dataset.loc[(scatter_dataset['EXITED'] == 1),column_neg] = np.NaN
            scatter_dataset.loc[(scatter_dataset['EXITED'] == 0),column_pos] = np.NaN
    
    return scatter_dataset


def creation_histo_full(test_dataset:pd.DataFrame):
    """This function creates the dataset for the histogram plot.  For every column (except Exited), histo_full will have a positive and negative version.
    The positive column will have NaN when the Exited is zero and the negative column will have NaN when the Exited is one. 

    Args:
        test_dataset (pd.DataFrame): the test dataset

    Returns:
        pd.DataFrame: the Dataframe used to display the Histogram
    """
    histo_full = test_dataset.copy()
    
    for column in histo_full.columns:
        column_neg = str(column)+'_neg'
        histo_full[column_neg] = histo_full[column]
        histo_full.loc[(histo_full['EXITED'] == 1),column_neg] = np.NaN
        histo_full.loc[(histo_full['EXITED'] == 0),column] = np.NaN
        
    return histo_full


def update_histogram_and_scatter(state):
    global x_selected, y_selected
    x_selected = state.x_selected
    y_selected = state.y_selected
    state.properties_scatter_dataset =  {"x":x_selected,
                                         "y[1]":y_selected+'_pos',
                                         "y[2]":y_selected+'_neg'} 
    state.scatter_dataset = state.scatter_dataset
    state.scatter_dataset_pred = state.scatter_dataset_pred

    state.properties_histo_full =  {"x[1]":x_selected,
                                    "x[2]":x_selected+'_neg'} 
    state.histo_full = state.histo_full
    state.histo_full_pred = state.histo_full_pred


dv_data_visualization_md = """
# Data **Visualization**{: .color-primary}
<|{dv_graph_selected}|toggle|lov={dv_graph_selector}|>

--------------------------------------------------------------------

<|part|render={dv_graph_selected == 'Histogram'}|
### Histogram
<|{x_selected}|selector|lov={select_x}|dropdown=True|label=Select x|>

<|{histo_full}|chart|type=histogram|properties={properties_histo_full}|rebuild|y=EXITED|label=EXITED|color[1]=red|color[2]=green|name[1]=Exited|name[2]=Stayed|height=600px|>
|>

<|part|render={dv_graph_selected == 'Scatter'}|
### Scatter
<|layout|columns= 1 2|
<|{x_selected}|selector|lov={select_x}|dropdown|label=Select x|>

<|{y_selected}|selector|lov={select_y}|dropdown|label=Select y|>
|>

<|{scatter_dataset}|chart|properties={properties_scatter_dataset}|rebuild|color[1]=red|color[2]=green|name[1]=Exited|name[2]=Stayed|mode=markers|type=scatter|height=600px|>
|>

"""

