_base_ = "pubtabnet.py"


train_dataloader = dict(dataset=dict(ann_file="~/data/mutab_pubtab250.pkl"))

val_dataloader = dict(dataset=dict(ann_file="~/data/mutab_pubtab250.pkl"))

test_dataloader = dict(dataset=dict(ann_file="~/data/mutab_pubtab250.pkl"))
