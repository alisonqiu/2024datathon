from pages.dialogs.dialog_roc_md import *

from pages.compare_models_md import *
from pages.data_visualization_md import *
from pages.databases_md import *
from pages.model_manager_md import *

# References and structure of all the dialogs

dialog_md = """
<|dialog|title=ROC Curve|partial={dialog_partial_roc}|open={dr_show_roc}|on_action=delete_dialog_roc|labels=Close|width=1000px|>
"""