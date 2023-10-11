from algos.algos import *
from taipy import Config, Scope
##############################################################################################################################
# Creation of the datanodes
##############################################################################################################################
# How to connect to the database
path_to_csv = 'data/churn.csv'

# path for csv and file_path for pickle
initial_dataset_cfg = Config.configure_data_node(id="initial_dataset",
                                             path=path_to_csv,
                                             storage_type="csv",
                                             has_header=True)

date_cfg = Config.configure_data_node(id="date", default_data="None")

preprocessed_dataset_cfg = Config.configure_data_node(id="preprocessed_dataset")

# the final datanode that contains the processed data
train_dataset_cfg = Config.configure_data_node(id="train_dataset")

# the final datanode that contains the processed data
trained_model_ml_cfg = Config.configure_data_node(id="trained_model_ml")
trained_model_baseline_cfg= Config.configure_data_node(id="trained_model_baseline")


# the final datanode that contains the processed data
test_dataset_cfg = Config.configure_data_node(id="test_dataset")

forecast_dataset_ml_cfg = Config.configure_data_node(id="forecast_dataset_ml")
forecast_dataset_baseline_cfg = Config.configure_data_node(id="forecast_dataset_baseline")

roc_data_ml_cfg = Config.configure_data_node(id="roc_data_ml")
roc_data_baseline_cfg = Config.configure_data_node(id="roc_data_baseline")

score_auc_ml_cfg = Config.configure_data_node(id="score_auc_ml")
score_auc_baseline_cfg = Config.configure_data_node(id="score_auc_baseline")


metrics_ml_cfg = Config.configure_data_node(id="metrics_ml")
metrics_baseline_cfg = Config.configure_data_node(id="metrics_baseline")

feature_importance_ml_cfg = Config.configure_data_node(id="feature_importance_ml")
feature_importance_baseline_cfg = Config.configure_data_node(id="feature_importance_baseline")

results_ml_cfg = Config.configure_data_node(id="results_ml")
results_baseline_cfg = Config.configure_data_node(id="results_baseline")


##############################################################################################################################
# Creation of the tasks
##############################################################################################################################

# the task will make the link between the input data node 
# and the output data node while executing the function

# initial_dataset --> preprocess dataset --> preprocessed_dataset
task_preprocess_dataset_cfg = Config.configure_task(id="preprocess_dataset",
                                                    input=[initial_dataset_cfg,date_cfg],
                                                    function=preprocess_dataset,
                                                    output=preprocessed_dataset_cfg)

# preprocessed_dataset --> create train data --> train_dataset, test_dataset
task_create_train_test_cfg = Config.configure_task(id="create_train_and_test_data",
                                                   input=preprocessed_dataset_cfg,
                                                   function=create_train_test_data,
                                                   output=[train_dataset_cfg, test_dataset_cfg])


# train_dataset --> create train_model data --> trained_model
task_train_model_baseline_cfg = Config.configure_task(id="train_model_baseline",
                                                      input=train_dataset_cfg,
                                                      function=train_model_baseline,
                                                      output=[trained_model_baseline_cfg,feature_importance_baseline_cfg])
        
# train_dataset --> create train_model data --> trained_model
task_train_model_ml_cfg = Config.configure_task(id="train_model_ml",
                                                input=train_dataset_cfg,
                                                function=train_model_ml,
                                                output=[trained_model_ml_cfg,feature_importance_ml_cfg])
                   

# test_dataset --> forecast --> forecast_dataset
task_forecast_baseline_cfg = Config.configure_task(id="predict_the_test_data_baseline",
                                                   input=[test_dataset_cfg, trained_model_baseline_cfg],
                                                   function=forecast,
                                                   output=forecast_dataset_baseline_cfg)
# test_dataset --> forecast --> forecast_dataset
task_forecast_ml_cfg = Config.configure_task(id="predict_the_test_data_ml",
                                             input=[test_dataset_cfg, trained_model_ml_cfg],
                                             function=forecast,
                                             output=forecast_dataset_ml_cfg)


task_roc_ml_cfg = Config.configure_task(id="task_roc_ml",
                                        input=[forecast_dataset_ml_cfg, test_dataset_cfg],
                                        function=roc_from_scratch,
                                        output=[roc_data_ml_cfg,score_auc_ml_cfg])

task_roc_baseline_cfg = Config.configure_task(id="task_roc_baseline",
                                              input=[forecast_dataset_baseline_cfg, test_dataset_cfg],
                                              function=roc_from_scratch,
                                              output=[roc_data_baseline_cfg,score_auc_baseline_cfg])


task_create_metrics_baseline_cfg = Config.configure_task(id="task_create_metrics_baseline",
                                                         input=[forecast_dataset_baseline_cfg,test_dataset_cfg],
                                                         function=create_metrics,
                                                         output=metrics_baseline_cfg)

task_create_metrics_ml_cfg = Config.configure_task(id="task_create_metrics",
                                                   input=[forecast_dataset_ml_cfg,test_dataset_cfg],
                                                   function=create_metrics,
                                                   output=metrics_ml_cfg)

task_create_results_baseline_cfg = Config.configure_task(id="task_create_results_baseline",
                                                         input=[forecast_dataset_baseline_cfg,test_dataset_cfg],
                                                         function=create_results,
                                                         output=results_baseline_cfg)

task_create_results_ml_cfg = Config.configure_task(id="task_create_results_ml",
                                                   input=[forecast_dataset_ml_cfg,test_dataset_cfg],
                                                   function=create_results,
                                                   output=results_ml_cfg)


##############################################################################################################################
# Creation of the scenario
##############################################################################################################################

scenario_cfg = Config.configure_scenario(id="churn_classification",
                                         task_configs=[task_create_metrics_baseline_cfg,
                                                       task_create_metrics_ml_cfg,
                                                       task_create_results_baseline_cfg,
                                                       task_create_results_ml_cfg,
                                                       task_forecast_baseline_cfg,
                                                       task_forecast_ml_cfg,
                                                       task_roc_ml_cfg,
                                                       task_roc_baseline_cfg,
                                                       task_train_model_baseline_cfg,
                                                       task_train_model_ml_cfg,
                                                       task_preprocess_dataset_cfg,
                                                       task_create_train_test_cfg])

Config.export('config/config.toml')