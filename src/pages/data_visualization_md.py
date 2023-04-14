import pandas as pd
import numpy as np


dv_graph_selector = ['Histogram','Scatter']
dv_graph_selected = dv_graph_selector[0]

# Histograms dialog
dv_width_histo = "100%"
dv_height_histo = 600

dv_dict_overlay = {'barmode':'overlay', "margin":{"t":20}}

dv_select_x_ = ['CREDITSCORE', 'AGE', 'TENURE', 'BALANCE', 'NUMOFPRODUCTS', 'HASCRCARD', 'ISACTIVEMEMBER', 'ESTIMATEDSALARY', 'GEOGRAPHY_FRANCE', 'GEOGRAPHY_GERMANY', 'GEOGRAPHY_SPAIN', 'GENDER_MALE']


def creation_scatter_dataset(test_dataset:pd.DataFrame):
    """This function creates the dataset for the scatter plot.  For every column (except Exited) will have a positive and negative version.
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


def update_histogram_and_scatter(column, state=None):
    if column == 'AGE' or column == 'CREDITSCORE' and state is not None:
        state.dv_dict_overlay = {'barmode':'overlay',"margin":{"t":20}}
    elif state is not None:
        state.dv_dict_overlay = {"margin":{"t":20}}

    if state is not None:
        state.properties_scatter_dataset =  {"x":column,
                                             "y[1]":state.y_selected+'_pos',
                                             "y[2]":state.y_selected+'_neg'} 
        state.scatter_dataset = state.scatter_dataset
        state.scatter_dataset_pred = state.scatter_dataset_pred

        state.properties_histo_full =  {"x[1]":column,
                                        "x[2]":column+'_neg'} 
        state.histo_full = state.histo_full
        state.histo_full_pred = state.histo_full_pred



def creation_histo_full(test_dataset:pd.DataFrame):
    """This function creates the dataset for the histogram plot.  For every column (except Exited) will have a positive and negative version.
    The positive column will have NaN when the Exited is zero and the negative column will have NaN when the Exited is one. 

    Args:
        test_dataset (pd.DataFrame): the test dataset

    Returns:
        pd.DataFrame: the Dataframe used to display the Histogram
    """
    histo_full = test_dataset.copy()
    # create a deterministic oversampling to have the same number of points for each class
    histo_1 = histo_full.loc[histo_full['EXITED'] == 1]    
    
    frames = [histo_full,histo_1,histo_1,histo_1]
    
    histo_full = pd.concat(frames, sort=False)
    
    for column in histo_full.columns:
        column_neg = str(column)+'_neg'
        histo_full[column_neg] = histo_full[column]
        histo_full.loc[(histo_full['EXITED'] == 1),column_neg] = np.NaN
        histo_full.loc[(histo_full['EXITED'] == 0),column] = np.NaN
        
    return histo_full

properties_histo_full = {}
properties_scatter_dataset = {}



dv_data_visualization_md = """
# Data Visualization

<|part|render={dv_graph_selected == 'Histogram'}|
<|layout|columns= 1 1 1|columns[mobile]=1|
Select type of graph : <br/> <|{dv_graph_selected}|selector|lov={dv_graph_selector}|dropdown|>

Select **x**: <br/>  <|{x_selected}|selector|lov={select_x}|dropdown=True|>
|>


<|{histo_full}|chart|type=histogram|properties={properties_histo_full}|rebuild|y=EXITED|label=EXITED|color[1]=red|color[2]=green|name[1]=Exited|name[2]=Stayed|height={dv_height_histo}|width={dv_width_histo}|layout={dv_dict_overlay}|class_name=histogram|>
|>

<|part|render={dv_graph_selected == 'Scatter'}|
<|layout|columns= 1 1 1|columns[mobile]=1|
Type of graph <br/> <|{dv_graph_selected}|selector|lov={dv_graph_selector}|dropdown|>

Select **x** <br/> <|{x_selected}|selector|lov={select_x}|dropdown=True|>

Select **y** <br/> <|{y_selected}|selector|lov={select_y}|dropdown=True|>
|>

<|{scatter_dataset}|chart|properties={properties_scatter_dataset}|rebuild|color[1]=red|color[2]=green|name[1]=Exited|name[2]=Stayed|height={dv_height_histo}|width={dv_width_histo}|mode=markers|type=scatter|layout={dv_dict_overlay}|>
|>

"""

