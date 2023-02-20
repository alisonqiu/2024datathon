# basic packages
import pandas as pd

# taipy functions
import taipy as tp
from taipy.gui import Gui, Icon, navigate

# get the config
from config.config import scenario_cfg
from taipy import Config

import os

# import to create the temporary file
import pathlib

# this path is used to create a temporary file that will allow us to
# download a table in the Datasouces page
tempdir = pathlib.Path(".tmp")
tempdir.mkdir(exist_ok=True)
PATH_TO_TABLE = str(tempdir / "table.csv")

###############################################################################
# we clean the data storage
###############################################################################

Config.configure_global_app(clean_entities_enabled=True)
tp.clean_all_entities()

##############################################################################################################################
# Execution of the scenario
##############################################################################################################################

tp.Core().run()

def create_first_scenario(scenario_cfg):
    global scenario
    scenario = tp.create_scenario(scenario_cfg)
    tp.submit(scenario)
    
create_first_scenario(scenario_cfg)

# ############################################################################################################################
# Initialization - Values from the scenario can be read
##############################################################################################################################
forecast_values_baseline = scenario.pipelines['pipeline_baseline'].forecast_dataset.read()
forecast_values = scenario.pipelines['pipeline_model'].forecast_dataset.read()

test_dataset = scenario.pipelines['pipeline_baseline'].test_dataset.read()
train_dataset = scenario.pipelines['pipeline_preprocessing'].train_dataset.read()
roc_dataset = scenario.pipelines['pipeline_baseline'].roc_data.read()

test_dataset.columns = [str(column).upper() for column in test_dataset.columns]

# it is for the data visualization with histogram and scatter plot
select_x = test_dataset.drop('EXITED',axis=1).columns.tolist()
x_selected = select_x[0]

select_y = select_x
y_selected = select_y[1]

##############################################################################################################################
# Initialization - Creation of a dataset that resume the results that will be used in a chart
##############################################################################################################################
from pages.main_dialog import *

values_baseline = scenario.pipelines['pipeline_baseline'].results.read()
values_model = scenario.pipelines['pipeline_model'].results.read()

values = values_baseline.copy()

forecast_series = values['Forecast']
true_series = values['Historical']


scatter_dataset_pred = creation_scatter_dataset_pred(test_dataset,forecast_series)
histo_full_pred = creation_histo_full_pred(test_dataset,forecast_series)

histo_full = creation_histo_full(test_dataset)
scatter_dataset = creation_scatter_dataset(test_dataset)

features_table = scenario.pipelines['pipeline_train_baseline'].feature_importance.read()

# Comparison of pipelines
# a generic code to take the correct pipelines
pipelines_to_compare = [pipeline for pipeline in scenario.pipelines if 'train' not in pipeline and 'preprocessing' not in pipeline]

accuracy_graph, f1_score_graph, score_auc_graph = compare_models_baseline(scenario, pipelines_to_compare) # comes from the compare_models.py

##############################################################################################################################
# Initialization - Creation of a pie chart to see the accuracy of the model that will be shown and also the distribution of the classes
##############################################################################################################################
# calculates the metrics for the 'baseline' model
(number_of_predictions,
 accuracy, f1_score, score_auc,
 number_of_good_predictions,
 number_of_false_predictions,
 fp_, tp_, fn_, tn_) = c_update_metrics(scenario, 'pipeline_baseline')

# pie charts
pie_plotly = pd.DataFrame({"values": [number_of_good_predictions, number_of_false_predictions],
                           "labels": ["Correct predictions", "False predictions"]})

distrib_class = pd.DataFrame({"values": [len(values[values["Historical"]==0]),len(values[values["Historical"]==1])],
                              "labels" : ["Stayed", "Exited"]})

##############################################################################################################################
# Initialization - Creation of the False/positive/negative/true table that will be shown
##############################################################################################################################

score_table = pd.DataFrame({"Score":["Predicted stayed", "Predicted exited"],
                            "Stayed": [tn_, fp_],
                            "Exited" : [fn_, tp_]})

pie_confusion_matrix = pd.DataFrame({"values": [tp_,tn_,fp_,fn_],
                              "labels" : ["True Positive","True Negative","False Positive",  "False Negative"]})

##############################################################################################################################
# Initialization - Creation of the graphical user interface (state)
##############################################################################################################################

# The list of pages that will be shown in the menu at the left of the page
menu_lov = [("Data Visualization", Icon('images/histogram_menu.svg', 'Data Visualization')),
            ("Model Manager", Icon('images/model.svg', 'Model Manager')),
            ("Compare Models", Icon('images/compare.svg', 'Compare Models')),
            ('Databases', Icon('images/Datanode.svg', 'Databases'))]

width_plotly = "450px"
height_plotly = "450px"

page_markdown = """
<|toggle|theme|>
<|menu|label=Menu|lov={menu_lov}|on_action=menu_fct|>
"""

# the initial page is the "Scenario Manager" page
page = "Data Visualization"
def menu_fct(state,var_name:str,fct,var_value):
    """Functions that is called when there is a change in the menu control

    Args:
        state : the state object of Taipy
        var_name (str): the changed variable name 
        var_value (obj): the changed variable value
    """
    # we change the value of the state.page variable in order to render the correct page
    try :
        state.page = var_value['args'][0]
        navigate(state, state.page.replace(" ", "-"))
    except:
        print("Warning : No args were found")
    pass


# Function for the prediction table. Bad predictions will be red and good predictions will be green (css class)
def get_style(state, index, row):
    return 'red' if row['Historical']!=row['Forecast'] else 'green'


##############################################################################################################################
# Creation of the entire markdown
##############################################################################################################################
pages = {
         "/":page_markdown+dialog_md,
         "Data-Visualization": dv_data_visualization_md,
         "Model-Manager": mm_model_manager_md, 
         "Compare-Models": cm_compare_models_md,
         "Databases": db_databases_md,}


# dialog_md is found in main_dialog.py
# the other are found in the dialogs folder
entire_markdown = page_markdown + dialog_md

# the object that will be used to generate the page
gui = Gui(pages=pages, css_file='main')
dialog_partial_roc = gui.add_partial(dialog_roc)

partial_scatter = gui.add_partial(creation_of_dialog_scatter(x_selected))
partial_histo = gui.add_partial(creation_of_dialog_histogram(x_selected))

partial_scatter_pred = gui.add_partial(creation_of_dialog_scatter_pred(x_selected))
partial_histo_pred = gui.add_partial(creation_of_dialog_histogram_pred(x_selected))

def update_partial_charts(state):
    """This function updates 4 partials containing charts and selectors. Partials are a mini-page 
    that can be reloaded in runtime with the functions below. They are reloaded in order to change the 
    content of the charts.

    Args:
        state: object containing all the variables used in the GUI
    """
    state.partial_scatter.update_content(state, creation_of_dialog_scatter(state.x_selected, state))
    state.partial_histo.update_content(state, creation_of_dialog_histogram(state.x_selected, state))
    
    state.partial_scatter_pred.update_content(state, creation_of_dialog_scatter_pred(state.x_selected, state))
    state.partial_histo_pred.update_content(state, creation_of_dialog_histogram_pred(state.x_selected, state))


##############################################################################################################################
# Updating displayed variables
##############################################################################################################################


def update_variables(state, pipeline):
    """This function updates the different variables and dataframes used in the application.

    Args:
        state: object containing all the variables used in the GUI
        pipeline (str): the name of the pipeline used to update the variables
    """
    global scenario
    pipeline_str = 'pipeline_'+pipeline
    
    if pipeline == 'baseline':
        state.values = scenario.pipelines[pipeline_str].results.read()
    else:
        state.values = scenario.pipelines[pipeline_str].results.read()
        
    state.forecast_series = state.values['Forecast']
    state.true_series = state.values["Historical"]
    
    
    (state.number_of_predictions,
    state.accuracy, state.f1_score, state.score_auc,
    number_of_good_predictions, number_of_false_predictions,
    fp_, tp_, fn_, tn_) = c_update_metrics(scenario, pipeline_str)
    
    
    update_charts(state, pipeline_str, number_of_good_predictions, number_of_false_predictions, fp_, tp_, fn_, tn_)
    


def update_charts(state, pipeline_str, number_of_good_predictions, number_of_false_predictions, fp_, tp_, fn_, tn_):
    """This function updates all the charts of the GUI.

    Args:
        state: object containing all the variables used in the GUI
        pipeline_str (str): the name of the pipeline shown
        number_of_good_predictions (int): number of good predictions
        number_of_false_predictions (int): number of false predictions
        fp_ (float): false positive rate
        tp_ (float): true positive rate
        fn_ (float): false negative rate
        tn_ (float): true negative rate
    """
    state.roc_dataset = scenario.pipelines[pipeline_str].roc_data.read()
    
    if 'model' in pipeline_str:
        state.features_table = scenario.pipelines['pipeline_train_model'].feature_importance.read()
    elif 'baseline' in pipeline_str:
        state.features_table = scenario.pipelines['pipeline_train_baseline'].feature_importance.read()

    state.score_table = pd.DataFrame({"Score":["Predicted stayed", "Predicted exited"],
                                      "Stayed": [tn_, fp_],
                                      "Exited" : [fn_, tp_]})

    state.pie_confusion_matrix = pd.DataFrame({"values": [tp_, tn_, fp_, fn_],
                                               "labels" : ["True Positive", "True Negative", "False Positive", "False Negative"]})

    state.scatter_dataset_pred = creation_scatter_dataset_pred(test_dataset, state.forecast_series)
    state.histo_full_pred = creation_histo_full_pred(test_dataset, state.forecast_series)

    
    # pie charts
    state.pie_plotly = pd.DataFrame({"values": [number_of_good_predictions, number_of_false_predictions],
                                     "labels": ["Correct predictions", "False predictions"]})

    state.distrib_class = pd.DataFrame({"values": [len(state.values[state.values["Historical"]==0]),
                                                   len(state.values[state.values["Historical"]==1])],
                                        "labels" : ["Stayed", "Exited"]})


##############################################################################################################################
# on_change function
##############################################################################################################################

# the other functions are in the right folder in frontend/dialogs
def on_change(state, var_name, var_value):
    """This function is called when a variable is changed in the GUI.

    Args:
        state : object containing all the variables used in the GUI
        var_name (str): name of the changed variable
        var_value (obj): value of the changed variable
    """
    if var_name == 'x_selected' or var_name == 'y_selected':
        update_partial_charts(state)
    
    if var_name == 'mm_algorithm_selected':
        if var_value == 'Baseline':
            update_variables(state,'baseline')
        if var_value == 'ML':
            update_variables(state,'model')
        
    if (var_name == 'mm_algorithm_selected' or var_name == "db_table_selected" and state.page == 'Databases') or (var_name == 'page' and var_value == 'Databases'):
        # if we are on the 'Databases' page, we have to create an temporary csv file
        handle_temp_csv_path(state)
           
    if var_name == 'page' and var_value != 'Databases':
        delete_temp_csv()



def delete_temp_csv():
    """This function deletes the temporary csv file."""
    if os.path.exists(PATH_TO_TABLE):
        os.remove(PATH_TO_TABLE)

def handle_temp_csv_path(state):
    """This function checks if the temporary csv file exists. If it does, it is deleted. Then, the temporary csv file
    is created for the right table

    Args:
        state: object containing all the variables used in the GUI
    """
    if os.path.exists(PATH_TO_TABLE):
        os.remove(PATH_TO_TABLE)
    if state.db_table_selected == 'Test Dataset':
        state.test_dataset.to_csv(PATH_TO_TABLE, sep=';')
    if state.db_table_selected == 'Confusion Matrix':
        state.score_table.to_csv(PATH_TO_TABLE, sep=';')
    if state.db_table_selected == "Training Dataset":
        train_dataset.to_csv(PATH_TO_TABLE, sep=';')
    if state.db_table_selected == "Forecast Dataset":
        values.to_csv(PATH_TO_TABLE, sep=';')
    

##############################################################################################################################
# Running the Gui
##############################################################################################################################
if __name__ == '__main__':
    gui.run(title="Churn classification",
    		host='0.0.0.0',
    		port=os.environ.get('PORT', '5050'),
    		dark_mode=False)
else:
    app = gui.run(title="Churn classification",
    	         dark_mode=False,
                 run_server=False)
