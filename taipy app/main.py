import pandas as pd
import taipy as tp
from taipy.gui import Gui, Icon, navigate
from config.config import scenario_cfg
from taipy.config import Config 
from pages.main_dialog import *

import warnings
with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=FutureWarning)

# Load configuration
Config.load('config/config.toml')
scenario_cfg = Config.scenarios['churn_classification']

# Execute the scenario
tp.Core().run()

def create_first_scenario(scenario_cfg):
    """Create and submit the first scenario."""
    scenario = tp.create_scenario(scenario_cfg)
    tp.submit(scenario)
    return scenario

scenario = create_first_scenario(scenario_cfg)

# Read datasets
train_dataset = scenario.train_dataset.read()
test_dataset = scenario.test_dataset.read()
dataset = scenario.initial_dataset.read()

# Process test dataset columns
# test_dataset.columns = [str(column).upper() for column in test_dataset.columns]

# Prepare data for visualization

select_x = dataset[['OilPeakRate',  'total_fluid', 'true_vertical_depth', 'frac_fluid_intensity', 'bin_lateral_length', 'gross_perforated_length_bins']].drop('OilPeakRate',axis=1).columns.tolist()
x_selected = select_x[0]
select_y = ['OilPeakRate']
y_selected = select_y[0]

# Read results and create charts
# values = scenario.results_ml.read()
# forecast_series = values['Forecast']
# scatter_dataset_pred = creation_scatter_dataset_pred(test_dataset, forecast_series)
# histo_full_pred = creation_histo_full_pred(test_dataset, forecast_series)
histo_full = creation_histo_full(dataset)
scatter_dataset = creation_scatter_dataset(dataset)
# features_table = scenario.RMSE_ml.read()
# accuracy_graph, f1_score_graph, score_auc_graph = compare_models_baseline(scenario, ['ml', 'baseline'])

def create_charts(model_type):
    """Create pie charts and metrics for the given model type."""
    metrics = c_update_metrics(scenario, model_type)
    (number_of_predictions, accuracy, f1_score, score_auc, 
     number_of_good_predictions, number_of_false_predictions, 
     fp_, tp_, fn_, tn_) = metrics
    
    pie_plotly = pd.DataFrame({
        "values": [number_of_good_predictions, number_of_false_predictions],
        "labels": ["Correct predictions", "False predictions"]
    })

    distrib_class = pd.DataFrame({
        "values": [len(values[values["Historical"]==0]), len(values[values["Historical"]==1])],
        "labels": ["Stayed", "OilPeakRate"]
    })

    score_table = pd.DataFrame({
        "Score": ["Predicted stayed", "Predicted OilPeakRate"],
        "Stayed": [tn_, fp_],
        "OilPeakRate": [fn_, tp_]
    })

    pie_confusion_matrix = pd.DataFrame({
        "values": [tp_, tn_, fp_, fn_],
        "labels": ["True Positive", "True Negative", "False Positive", "False Negative"]
    })

    return (number_of_predictions, number_of_false_predictions, number_of_good_predictions, 
            accuracy, f1_score, score_auc, pie_plotly, distrib_class, score_table, pie_confusion_matrix)

# # Initialize charts
# chart_metrics = create_charts('ml')
# (number_of_predictions, number_of_false_predictions, number_of_good_predictions, 
#  accuracy, f1_score, score_auc, pie_plotly, distrib_class, score_table, pie_confusion_matrix) = chart_metrics

def on_change(state, var_name, var_value):
    """Handle variable changes in the GUI."""
    if var_name in ['x_selected', 'y_selected']:
        update_histogram_and_scatter(state)
    elif var_name == 'mm_algorithm_selected':
        # update_variables(state, var_value.lower()) TODO: update
        update_histogram_and_scatter(state)
    elif var_name in ['mm_algorithm_selected', 'db_table_selected']:
        handle_temp_csv_path(state)
        # update_histogram_and_scatter(state)

# GUI initialization
menu_lov = [
    ("Data Visualization", Icon('images/histogram_menu.svg', 'Data Visualization')),
    # ("Model Manager", Icon('images/model.svg', 'Model Manager')),
    # ("Compare Models", Icon('images/compare.svg', 'Compare Models')),
    ('Databases', Icon('images/Datanode.svg', 'Databases'))
]

root_md = """
<|toggle|theme|>
<|menu|label=Menu|lov={menu_lov}|on_action=menu_fct|>
"""

page = "Data Visualization"

def menu_fct(state, var_name, var_value):
    """Function that is called when there is a change in the menu control."""
    state.page = var_value['args'][0]
    navigate(state, state.page.replace(" ", "-"))

def update_variables(state, model_type):
    """Update the different variables and dataframes used in the application."""
    global scenario
    state.values = scenario.data_nodes[f'results_{model_type}'].read()
    state.forecast_series = state.values['Forecast']
    
    metrics = c_update_metrics(scenario, model_type)
    (state.number_of_predictions, state.accuracy, state.f1_score, state.score_auc,
     number_of_good_predictions, number_of_false_predictions, fp_, tp_, fn_, tn_) = metrics
    
    update_charts(state, model_type, number_of_good_predictions, number_of_false_predictions, fp_, tp_, fn_, tn_)

def update_charts(state, model_type, number_of_good_predictions, number_of_false_predictions, fp_, tp_, fn_, tn_):
    """This function updates all the charts of the GUI.

    Args:
        state: object containing all the variables used in the GUI
        model_type (str): the name of the model_type shown
        number_of_good_predictions (int): number of good predictions
        number_of_false_predictions (int): number of false predictions
        fp_ (float): false positive rate
        tp_ (float): true positive rate
        fn_ (float): false negative rate
        tn_ (float): true negative rate
    """
    state.roc_dataset = scenario.data_nodes[f'roc_data_{model_type}'].read()
    state.features_table = scenario.data_nodes[f'RMSE_{model_type}'].read()

    state.score_table = pd.DataFrame({"Score":["Predicted stayed", "Predicted OilPeakRate"],
                                      "Stayed": [tn_, fp_],
                                      "OilPeakRate" : [fn_, tp_]})

    state.pie_confusion_matrix = pd.DataFrame({"values": [tp_, tn_, fp_, fn_],
                                               "labels" : ["True Positive", "True Negative", "False Positive", "False Negative"]})

    state.scatter_dataset_pred = creation_scatter_dataset_pred(test_dataset, state.forecast_series)
    state.histo_full_pred = creation_histo_full_pred(test_dataset, state.forecast_series)
    
    # pie charts
    state.pie_plotly = pd.DataFrame({"values": [number_of_good_predictions, number_of_false_predictions],
                                     "labels": ["Correct predictions", "False predictions"]})

    state.distrib_class = pd.DataFrame({"values": [len(state.values[state.values["Historical"]==0]),
                                                   len(state.values[state.values["Historical"]==1])],
                                        "labels" : ["Stayed", "OilPeakRate"]})

def on_init(state):
    update_histogram_and_scatter(state)

# Define pages
pages = {
    "/": root_md,
    "Data-Visualization": dv_data_visualization_md,
    # "Model-Manager": mm_model_manager_md, 
    # "Compare-Models": cm_compare_models_md,
    "Databases": db_databases_md,
}

# Run the GUI
if __name__ == '__main__':
    gui = Gui(pages=pages)
    gui.run(title="Churn classification",  dark_mode=False, port=8494)
