# Confusion matrix dialog

show_fp = False

def show_fp_fct(gui):
    gui.show_fp = True   
def delete_dialog_fp(gui):
    gui.show_fp = False

dialog_table_fp = """
<center>
<|{score_table}|table|width=800px|height=200px|width=450px|>
</center>
"""