# Selector to select the table to show
db_table_selector = ['Training Dataset', 'Test Dataset', 'Forecast Dataset', 'Confusion Matrix']
db_table_selected = db_table_selector[0]

# Aggregation of the strings to create the complete page
db_databases_md = """
# Databases

<|layout|columns=2 2 1|columns[mobile]=1|
<|{mm_algorithm_selected}|selector|lov={mm_algorithm_selector}|dropdown|label=Algorithm|>

<|{db_table_selected}|selector|lov={db_table_selector}|dropdown|label=Table|>

<|{PATH_TO_TABLE}|file_download|name=table.csv|label=Download table|>
|>

<Confusion|part|render={db_table_selected=='Confusion Matrix'}|
<|{score_table}|table|width=fit-content|show_all|class_name=ml-auto mr-auto|>
|Confusion>

<Training|part|render={db_table_selected=='Training Dataset'}|
<|{train_dataset}|table|width=100%|>
|Training>

<Forecast|part|render={db_table_selected=='Forecast Dataset'}|
<|{values}|table|width=fit-content|style={lambda s,i,r: 'red_color' if r['Historical']!=r['Forecast'] else 'green_color'}|class_name=ml-auto mr-auto|>
|Forecast>

<test_dataset|part|render={db_table_selected=='Test Dataset'}|
<|{test_dataset}|table|width=100%|>
|test_dataset>
""" 

