import pandas as pd

# taipy functions
import taipy as tp
from taipy.gui import Gui, Icon, navigate

# get the config
from config.config import scenario_cfg

##############################################################################################################################
# Execution of the scenario
##############################################################################################################################

tp.Core().run()

def create_first_scenario(scenario_cfg):
    scenario = tp.create_scenario(scenario_cfg)
    tp.submit(scenario)
    return scenario
    
scenario = create_first_scenario(scenario_cfg)

#############################################################################################################################
# Initialization - Values from the scenario can be read
##############################################################################################################################
train_dataset = scenario.train_dataset.read()
test_dataset = scenario.test_dataset.read()
roc_dataset = scenario.pipeline_model.roc_data.read()

test_dataset.columns = [str(column).upper() for column in test_dataset.columns]

# it is for the data visualization with histogram and scatter plot
select_x = test_dataset.drop('EXITED',axis=1).columns.tolist()
x_selected = select_x[0]

select_y = select_x
y_selected = select_y[1]

##############################################################################################################################
# Initialization - Creation of the charts
##############################################################################################################################
from pages.main_dialog import *

values = scenario.pipeline_model.results.read()

forecast_series = values['Forecast']

scatter_dataset_pred = creation_scatter_dataset_pred(test_dataset, forecast_series)
histo_full_pred = creation_histo_full_pred(test_dataset, forecast_series)

histo_full = creation_histo_full(test_dataset)
scatter_dataset = creation_scatter_dataset(test_dataset)

features_table = scenario.pipeline_model.feature_importance.read()

# Comparison of pipelines
# a generic code to take the correct pipelines
accuracy_graph, f1_score_graph, score_auc_graph = compare_models_baseline(scenario, scenario.pipelines) # comes from the compare_models.py

##############################################################################################################################
# Initialization - Creation of a pie chart to see the accuracy of the model that will be shown and also the distribution of the classes
##############################################################################################################################
def create_charts(pipeline_str): 
    (number_of_predictions,
    accuracy, f1_score, score_auc,
    number_of_good_predictions,
    number_of_false_predictions,
    fp_, tp_, fn_, tn_) = c_update_metrics(scenario, pipeline_str)

    # pie charts
    pie_plotly = pd.DataFrame({"values": [number_of_good_predictions, number_of_false_predictions],
                            "labels": ["Correct predictions", "False predictions"]})

    distrib_class = pd.DataFrame({"values": [len(values[values["Historical"]==0]), len(values[values["Historical"]==1])],
                                  "labels" : ["Stayed", "Exited"]})

    score_table = pd.DataFrame({"Score":["Predicted stayed", "Predicted exited"],
                                "Stayed": [tn_, fp_],
                                "Exited" : [fn_, tp_]})

    pie_confusion_matrix = pd.DataFrame({"values": [tp_,tn_,fp_,fn_],
                                         "labels" : ["True Positive", "True Negative",
                                                     "False Positive",  "False Negative"]})

    return number_of_predictions, number_of_false_predictions, number_of_good_predictions, accuracy, f1_score, score_auc, pie_plotly, distrib_class, score_table, pie_confusion_matrix


number_of_predictions, number_of_false_predictions, number_of_good_predictions,\
accuracy, f1_score, score_auc,\
pie_plotly, distrib_class, score_table, pie_confusion_matrix = create_charts('pipeline_model')

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
        update_histogram_and_scatter(state)

    if var_name == 'mm_algorithm_selected':
        if var_value == 'Baseline':
            update_variables(state,'baseline')
        if var_value == 'ML':
            update_variables(state,'model')

    if var_name == 'mm_algorithm_selected' or var_name == "db_table_selected":
        # if we are on the 'Databases' page, we have to create an temporary csv file
        handle_temp_csv_path(state)


##############################################################################################################################
# Initialization - Creation of the graphical user interface (state)
##############################################################################################################################

# The list of pages that will be shown in the menu at the left of the page
menu_lov = [("Data Visualization", Icon('images/histogram_menu.svg', 'Data Visualization')),
            ("Model Manager", Icon('images/model.svg', 'Model Manager')),
            ("Compare Models", Icon('images/compare.svg', 'Compare Models')),
            ('Databases', Icon('images/Datanode.svg', 'Databases'))]

root_md = """
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
    state.page = var_value['args'][0]
    navigate(state, state.page.replace(" ", "-"))

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
    state.features_table = scenario.pipelines[pipeline_str].feature_importance.read()

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
# Creation of the entire markdown
##############################################################################################################################
pages = {"/":root_md + dialog_md,
         "Data-Visualization": dv_data_visualization_md,
         "Model-Manager": mm_model_manager_md, 
         "Compare-Models": cm_compare_models_md,
         "Databases": db_databases_md,}

def on_init(state):
    update_histogram_and_scatter(state)

##############################################################################################################################
# Running the Gui
##############################################################################################################################
if __name__ == '__main__':
    gui = Gui(pages=pages)
    gui.run(title="Churn classification", dark_mode=False)
