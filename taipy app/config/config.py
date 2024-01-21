from algos.algos import *
from taipy import Config, Scope
##############################################################################################################################
# Creation of the datanodes
##############################################################################################################################
# How to connect to the database
path_to_csv = 'data/cleaned.csv'

# path for csv and file_path for pickle
initial_dataset_cfg = Config.configure_data_node(id="initial_dataset",
                                             path=path_to_csv,
                                             storage_type="csv",
                                             has_header=True) #need


# the final datanode that contains the processed data
train_dataset_cfg = Config.configure_data_node(id="train_dataset")
test_dataset_cfg = Config.configure_data_node(id="test_dataset")


train_dataset_x_cfg = Config.configure_data_node(id="train_dataset_x")#need
train_dataset_y_cfg = Config.configure_data_node(id="train_dataset_y")

# the final datanode that contains the processed data
trained_model_ml_cfg = Config.configure_data_node(id="trained_model_ml")
trained_model_baseline_cfg= Config.configure_data_node(id="trained_model_baseline")


# the final datanode that contains the processed data
test_dataset_x_cfg = Config.configure_data_node(id="test_dataset_x")#need
test_dataset_y_cfg = Config.configure_data_node(id="test_dataset_y")

forecast_dataset_ml_cfg = Config.configure_data_node(id="forecast_dataset_ml")#need
forecast_dataset_baseline_cfg = Config.configure_data_node(id="forecast_dataset_baseline")#need

RMSE_ml_cfg = Config.configure_data_node(id="RMSE_ml")#need: RMSE_ml
RMSE_baseline_cfg = Config.configure_data_node(id="RMSE_baseline")#need: RMSE_baseline


##############################################################################################################################
# Creation of the tasks
##############################################################################################################################

# the task will make the link between the input data node 
# and the output data node while executing the function

# preprocessed_dataset --> create train data --> train_dataset, test_dataset
task_create_train_test_cfg = Config.configure_task(id="create_train_and_test_data",
                                                   input=initial_dataset_cfg, #need:  input=initial_dataset_cfg
                                                   function=create_train_test_data,
                                                   output=[train_dataset_x_cfg, test_dataset_x_cfg, train_dataset_y_cfg, test_dataset_y_cfg, train_dataset_cfg, test_dataset_cfg])


# train_dataset --> create train_model data --> trained_model
task_train_model_baseline_cfg = Config.configure_task(id="train_model_baseline",
                                                      input=[train_dataset_x_cfg, test_dataset_x_cfg, train_dataset_y_cfg, test_dataset_y_cfg],
                                                      function=train_model_baseline, #need
                                                      output=[trained_model_baseline_cfg,RMSE_baseline_cfg, forecast_dataset_ml_cfg]) #need: output=[trained_model_baseline_cfg, RMSE_baseline_cfg,forecast_dataset_ml_cfg]
        
# train_dataset --> create train_model data --> trained_model
task_train_model_ml_cfg = Config.configure_task(id="train_model_ml",
                                                input=[train_dataset_x_cfg, test_dataset_x_cfg, train_dataset_y_cfg, test_dataset_y_cfg],
                                                function=train_model_ml,
                                                output=[trained_model_ml_cfg,RMSE_ml_cfg,forecast_dataset_baseline_cfg])#need: output=[trained_model_ml_cfg, RMSE_ml_cfg,forecast_dataset_baseline_cfg]
                   

##############################################################################################################################
# Creation of the scenario
##############################################################################################################################

scenario_cfg = Config.configure_scenario(id="churn_classification",
                                         task_configs=[
                                                       task_train_model_baseline_cfg, #need
                                                       task_train_model_ml_cfg, #need
                                                       task_create_train_test_cfg]) #need

Config.export('config/config.toml')