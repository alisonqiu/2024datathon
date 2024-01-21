import pathlib

# This path is used to create a temporary CSV file download the table
tempdir = pathlib.Path(".tmp")
tempdir.mkdir(exist_ok=True)
PATH_TO_TABLE = str(tempdir / "table.csv")
chev = "../images/chevron.png"
chev_label = "../images/label.png"

# Selector to select the table to show
db_table_selector = ['Training Dataset', 'Test Dataset']
db_table_selected = db_table_selector[0]

def handle_temp_csv_path(state):
    """This function checks if the temporary csv file exists. If it does, it is deleted. Then, the temporary csv file
    is created for the right table

    Args:
        state: object containing all the variables used in the GUI
    """
    if state.db_table_selected == 'Test Dataset':
        state.test_dataset.to_csv(PATH_TO_TABLE, sep=';')
    if state.db_table_selected == "Training Dataset":
        state.train_dataset.to_csv(PATH_TO_TABLE, sep=';')



# Aggregation of the strings to create the complete page
db_databases_md = """
<center><|{chev}|image|id=biglogo|></center>
#  <div align="center"><|{chev_label}|image|>  Database  <|{chev_label}|image|></div>


<|layout|columns=2 2 1|
<|{mm_algorithm_selected}|selector|lov={mm_algorithm_selector}|dropdown|label=Algorithm|active=False|>

<|{db_table_selected}|selector|lov={db_table_selector}|dropdown|label=Table|>

<|{PATH_TO_TABLE}|file_download|name=table.csv|label=Download table|>
|>


<Training|part|render={db_table_selected=='Training Dataset'}|
<|{train_dataset}|table|>
|Training>


<test_dataset|part|render={db_table_selected=='Test Dataset'}|
<|{test_dataset}|table|>
|test_dataset>
""" 

