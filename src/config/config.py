from algos.algos import *
from taipy import Scope, Frequency, Config
##############################################################################################################################
# Creation of the datanodes
##############################################################################################################################
# How to connect to the database
path_to_csv = 'data/churn.csv'

# path for csv and file_path for pickle
initial_dataset = Config.configure_data_node(id="initial_dataset",
                                             path=path_to_csv,
                                             storage_type="csv",
                                             has_header=True)

date_cfg = Config.configure_data_node(id="date", default_data="None")

preprocessed_dataset = Config.configure_data_node(id="preprocessed_dataset",
                                                  cacheable=True,
                                                  validity_period=dt.timedelta(days=1))

# the final datanode that contains the processed data
train_dataset = Config.configure_data_node(id="train_dataset",
                                           cacheable=True,
                                           validity_period=dt.timedelta(days=1))

# the final datanode that contains the processed data
trained_model = Config.configure_data_node(id="trained_model",
                                           cacheable=True,
                                           validity_period=dt.timedelta(days=1))

trained_model_baseline = Config.configure_data_node(id="trained_model_baseline",
                                                    cacheable=True,
                                                    validity_period=dt.timedelta(days=1))


# the final datanode that contains the processed data
test_dataset = Config.configure_data_node(id="test_dataset",
                                          cacheable=True,
                                          validity_period=dt.timedelta(days=1))

forecast_dataset = Config.configure_data_node(id="forecast_dataset",
                                          scope=Scope.PIPELINE,
                                             cacheable=True,
                                             validity_period=dt.timedelta(days=1))

roc_data = Config.configure_data_node(id="roc_data",
                                      scope=Scope.PIPELINE,
                                      cacheable=True,
                                      validity_period=dt.timedelta(days=1))

score_auc = Config.configure_data_node(id="score_auc",
                                       scope=Scope.PIPELINE,
                                       cacheable=True,
                                       validity_period=dt.timedelta(days=1))

metrics = Config.configure_data_node(id="metrics",
                                     scope=Scope.PIPELINE,
                                     cacheable=True,
                                     validity_period=dt.timedelta(days=1))

feature_importance_cfg = Config.configure_data_node(id="feature_importance",
                                                    scope=Scope.PIPELINE)


results = Config.configure_data_node(id="results",
                                     scope=Scope.PIPELINE,
                                     cacheable=True,
                                     validity_period=dt.timedelta(days=1))


##############################################################################################################################
# Creation of the tasks
##############################################################################################################################

# the task will make the link between the input data node 
# and the output data node while executing the function

# initial_dataset --> preprocess dataset --> preprocessed_dataset
task_preprocess_dataset = Config.configure_task(id="preprocess_dataset",
                                                input=[initial_dataset,date_cfg],
                                                function=preprocess_dataset,
                                                output=preprocessed_dataset)

# preprocessed_dataset --> create train data --> train_dataset, test_dataset
task_create_train_test = Config.configure_task(id="create_train_and_test_data",
                                               input=preprocessed_dataset,
                                               function=create_train_test_data,
                                               output=[train_dataset, test_dataset])


# train_dataset --> create train_model data --> trained_model
task_train_model = Config.configure_task(id="train_model",
                                         input=train_dataset,
                                         function=train_model,
                                         output=[trained_model,feature_importance_cfg])
                                   
# train_dataset --> create train_model data --> trained_model
task_train_model_baseline = Config.configure_task(id="train_model_baseline",
                                                  input=train_dataset,
                                                  function=train_model_baseline,
                                                  output=[trained_model_baseline,feature_importance_cfg])

# test_dataset --> forecast --> forecast_dataset
task_forecast = Config.configure_task(id="predict_the_test_data",
                                      input=[test_dataset, trained_model],
                                      function=forecast,
                                      output=forecast_dataset)

# test_dataset --> forecast --> forecast_dataset
task_forecast_baseline = Config.configure_task(id="predict_of_baseline",
                           input=[test_dataset, trained_model_baseline],
                           function=forecast_baseline,
                           output=forecast_dataset)

task_roc = Config.configure_task(id="task_roc",
                           input=[forecast_dataset, test_dataset],
                           function=roc_from_scratch,
                           output=[roc_data,score_auc])

task_roc_baseline = Config.configure_task(id="task_roc_baseline",
                           input=[forecast_dataset, test_dataset],
                           function=roc_from_scratch,
                           output=[roc_data,score_auc])

task_create_metrics = Config.configure_task(id="task_create_metrics",
                                            input=[forecast_dataset,test_dataset],
                                            function=create_metrics,
                                            output=metrics)

task_create_results = Config.configure_task(id="task_create_results",
                                            input=[forecast_dataset,test_dataset],
                                            function=create_results,
                                            output=results)



##############################################################################################################################
# Creation of the pipeline and the scenario
##############################################################################################################################

# configuration of the pipeline and scenario
pipeline_preprocessing = Config.configure_pipeline(id="pipeline_preprocessing", task_configs=[task_preprocess_dataset,
                                                                                              task_create_train_test])

pipeline_train_baseline = Config.configure_pipeline(id="pipeline_train_baseline", task_configs=[task_train_model_baseline])
pipeline_train_model = Config.configure_pipeline(id="pipeline_train_model", task_configs=[task_train_model])

pipeline_model = Config.configure_pipeline(id="pipeline_model", task_configs=[task_forecast,
                                                                              task_roc,
                                                                              task_create_metrics,
                                                                              task_create_results])

pipeline_baseline = Config.configure_pipeline(id="pipeline_baseline", task_configs=[task_forecast_baseline,
                                                                                    task_roc_baseline,
                                                                                    task_create_metrics,
                                                                                    task_create_results])

# the scenario will run the pipelines
scenario_cfg = Config.configure_scenario(id="churn_classification",
                                         pipeline_configs=[pipeline_preprocessing, 
                                                           pipeline_train_baseline, pipeline_train_model,
                                                           pipeline_model,pipeline_baseline],
                                         frequency=Frequency.WEEKLY)

































