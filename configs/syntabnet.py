_base_ = "pubtabnet.py"


model = dict(
    handler=dict(
        html_dict=dict(load="alphabet/syntabnet/structure_alphabet.txt"),
        cell_dict=dict(load="alphabet/syntabnet/character_alphabet.txt"),
    )
)

train_dataloader = dict(dataset=dict(ann_file="~/data/mutab_syntabnet.pkl"))

val_dataloader = dict(dataset=dict(ann_file="~/data/mutab_syntabnet.pkl"))

test_dataloader = dict(dataset=dict(ann_file="~/data/mutab_syntabnet.pkl"))
