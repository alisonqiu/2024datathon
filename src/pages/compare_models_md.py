import numpy as np

# See what is normally done in this page in the Develop branch

def c_update_metrics(scenario, pipeline):
    """This function updates the metrics of a scenario using a pipeline

    Args:
        scenario (scenario): the selected scenario
        pipeline (str): the name of the selected pipeline

    Returns:
        obj: a number of values, lists that represent the metrics
    """
    metrics = scenario.pipelines[pipeline].metrics.read()

    number_of_predictions = metrics['number_of_predictions']
    number_of_good_predictions = metrics['number_of_good_predictions']
    number_of_false_predictions = metrics['number_of_false_predictions']

    accuracy = np.around(metrics['accuracy'], decimals=2)
    f1_score = np.around(metrics['f1_score'], decimals=2)
    score_auc = np.around(scenario.pipelines[pipeline].score_auc.read(), decimals=2)
    
    dict_ftpn = metrics['dict_ftpn']
    
    fp_ = dict_ftpn['fp']
    tp_ = dict_ftpn['tp']
    fn_ = dict_ftpn['fn']
    tn_ = dict_ftpn['tn']
    return number_of_predictions, accuracy, f1_score, score_auc, number_of_good_predictions, number_of_false_predictions, fp_, tp_, fn_, tn_

