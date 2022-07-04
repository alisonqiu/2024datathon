# Roc Dialog
dr_show_roc = False

def show_roc_fct(state):
    state.dr_show_roc = True  
def delete_dialog_roc(state):
    state.dr_show_roc = False

dialog_roc = """
<center>
<|{roc_dataset}|chart|x=False positive rate|y[1]=True positive rate|label[1]=True positive rate|height=500px|width=900px|type=scatter|>
</center>
"""