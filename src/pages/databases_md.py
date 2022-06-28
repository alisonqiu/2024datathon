# Confusion matrix dialog
db_confusion_matrix_md = """
<|part|render={db_table_selected=='Confusion Matrix'}|
<center>
<|{score_table}|table|height=200px|width=400px|show_all=True|>
</center>
|>
"""

db_train_dataset_md = """
<|part|render={db_table_selected=='Training Dataset'}|
<|{train_dataset}|table|width=1400px|height=560px|>
|>
"""

db_forecast_dataset_md = """
<|part|render={db_table_selected=='Forecast Dataset'}|
<center>
<|{values}|table|height=560px|width=800px|style={get_style}|>
</center>
|>
"""


db_test_dataset_md = """
<|part|render={db_table_selected=='Test Dataset'}|
<|{test_dataset}|table|width=1400px|height=560px|>
|>
"""


db_table_selector = ['Training Dataset', 'Test Dataset', 'Forecast Dataset', 'Confusion Matrix']
db_table_selected = db_table_selector[0]

db_databases_md = """
# Databases

<|layout|columns=2 2 1|columns[mobile]=1|
<|
**Algorithm**: \n \n <|{mm_algorithm_selected}|selector|lov={mm_algorithm_selector}|dropdown=True|>
|>

<|
**Table**: \n \n <|{db_table_selected}|selector|lov={db_table_selector}|dropdown=True|>
|>

<br/>
<br/>
<|{PATH_TO_TABLE}|file_download|name=table.csv|label=Download table|>
|>
""" + db_test_dataset_md + db_confusion_matrix_md + db_train_dataset_md + db_forecast_dataset_md


