import pandas as pd
import numpy as np


dv_graph_selector = ['Histogram','Scatter Plot','Heatmap']
dv_graph_selected = dv_graph_selector[0]
chev = "../images/chevron.png"
chev_label = "../images/label.png"

# Histograms dialog
properties_histo_full = {}
properties_scatter_dataset = {}

def creation_scatter_dataset(test_dataset:pd.DataFrame):
    """This function creates the dataset for the scatter plot.  For every column (except OilPeakRate), scatter_dataset will have a positive and negative version.
    The positive column will have NaN when the OilPeakRate is zero and the negative column will have NaN when the OilPeakRate is one.

    Args:
        test_dataset (pd.DataFrame): the test dataset

    Returns:
        pd.DataFrame: the datafram
    """
    scatter_dataset = test_dataset.copy()

    # for column in scatter_dataset.columns:
    #     if column != 'OilPeakRate' :
    #         column_neg = str(column)+'_neg'
    #         column_pos = str(column)+'_pos'
            
    #         scatter_dataset[column_neg] = scatter_dataset[column]
    #         scatter_dataset[column_pos] = scatter_dataset[column]
            
    #         scatter_dataset.loc[(scatter_dataset['OilPeakRate'] == 1),column_neg] = np.NaN
    #         scatter_dataset.loc[(scatter_dataset['OilPeakRate'] == 0),column_pos] = np.NaN
    
    return scatter_dataset


def creation_histo_full(test_dataset:pd.DataFrame):
    """This function creates the dataset for the histogram plot.  For every column (except OilPeakRate), histo_full will have a positive and negative version.
    The positive column will have NaN when the OilPeakRate is zero and the negative column will have NaN when the OilPeakRate is one. 

    Args:
        test_dataset (pd.DataFrame): the test dataset

    Returns:
        pd.DataFrame: the Dataframe used to display the Histogram
    """
    histo_full = test_dataset.copy()
    
    # for column in histo_full.columns:
    #     column_neg = str(column)+'_neg'
    #     histo_full[column_neg] = histo_full[column]
    #     histo_full.loc[(histo_full['OilPeakRate'] == 1),column_neg] = np.NaN
    #     histo_full.loc[(histo_full['OilPeakRate'] == 0),column] = np.NaN
        
    return histo_full


def update_histogram_and_scatter(state):
    global x_selected, y_selected
    x_selected = state.x_selected
    y_selected = state.y_selected
    y_selected2 = state.y_selected
    state.properties_scatter_dataset =  {"x":x_selected,
                                         "y":y_selected,} 
    state.scatter_dataset = state.scatter_dataset
    # state.scatter_dataset_pred = state.scatter_dataset_pred

    state.properties_histo_full =  {"x[1]":x_selected} 
    state.histo_full = state.histo_full
    # state.histo_full_pred = state.histo_full_pred

data = {
    "correlation" : [
    [1.000000, 0.408061, 0.335049, 0.320376, 0.296047, 0.336264],
    [0.408061, 1.000000, 0.677548, 0.621770, 0.125492, 0.819550],
    [0.335049, 0.677548, 1.000000, 0.960729, 0.004759, 0.337680],
    [0.320376, 0.621770, 0.960729, 1.000000, 0.026506, 0.302237],
    [0.296047, 0.125492, 0.004759, 0.026506, 1.000000, 0.245397],
    [0.336264, 0.819550, 0.337680, 0.302237, 0.245397, 1.000000]
],
    "x": ["OilPeakRate", "total_fluid", "gross_perforated_length", "bin_lateral_length", "true_vertical_depth", "frac_fluid_intensity"],
    "y": ["OilPeakRate", "total_fluid", "gross_perforated_length", "bin_lateral_length", "true_vertical_depth",	"frac_fluid_intensity"]
}

layout = {
    # This array contains the information we want to display in the cells
    # These are filled later
    "annotations": [],
    # No ticks on the x axis, show labels on top the of the chart
    "xaxis": {
        "ticks": "",
        "side": "top"
    },
    # No ticks on the y axis
    # Add a space character for a small margin with the text
    "yaxis": {
        "ticks": "",
        "ticksuffix": " "
    },
    "margin":{"t":-50}
}

# Iterate over all cities
for i in range(6):
    # Iterate over all seasons
    for j in range(6):
        corr = data["correlation"][i][j]
        # Create the annotation
        annotation = {
            # The name of the season
            "x": i,
            # The name of the city
            "y": j,
            # The temperature, as a formatted string
            "text": f"{corr}",
            "font": {
                "color": "white" 
            },
            # Remove the annotation arrow
            "showarrow": False
        }
        # Add the annotation to the layout's annotations array
        layout["annotations"].append(annotation)
options = { "colorscale": "Portland" }

dv_data_visualization_md = """
<center><|{chev}|image|id=biglogo|></center>
#  <div align="center"><|{chev_label}|image|> Data Visualization   <|{chev_label}|image|></div>

<center>
<|{dv_graph_selected}|toggle|lov={dv_graph_selector}|>
</center>
--------------------------------------------------------------------

<|part|render={dv_graph_selected == 'Histogram'}|
###Histogram
<|{x_selected}|selector|lov={select_x}|dropdown=True|label=Select x|>

<|{histo_full}|chart|type=histogram|properties={properties_histo_full}|rebuild|y=OilPeakRate|label=OilPeakRate|color[1]={"#f15a69"}|height=600px|>
|>

<|part|render={dv_graph_selected == 'Scatter Plot'}|
### Scatter
<|layout|columns= 1 2 1|
<|{x_selected}|selector|lov={select_x}|dropdown|label=Select x|>

<|{y_selected}|selector|lov={select_y}|dropdown|label=Select y|>
|>

<|{scatter_dataset}|chart|properties={properties_scatter_dataset}|rebuild|color={"#2FA2CE"}|name[1]=OilPeakRate|mode=markers|type=scatter|height=600px|>
|>

<|part|render={dv_graph_selected == 'Heatmap'}|
### Heatmap
<center>
<|{data}|chart|type=heatmap|z=correlation|x=x|y=y|layout={layout}|title={"Correlation Heatmap of Top Variables with OilPeakRate"}|options={options}|labels=None|>
</center>
"""

