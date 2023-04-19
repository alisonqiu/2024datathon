from pages.compare_models_md import *
from pages.data_visualization_md import *
from pages.databases_md import *
from pages.model_manager_md import *

dr_show_roc = False

dialog_md = """
<|dialog|open={dr_show_roc}|title=ROC Curve|on_action={lambda s: s.assign("dr_show_roc", False)}|labels=Close|width=1000px|
<|{roc_dataset}|chart|x=False positive rate|y[1]=True positive rate|label[1]=True positive rate|height=500px|width=900px|type=scatter|>
|>
"""