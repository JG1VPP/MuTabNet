_base_ = "pubtabnet.py"


model = dict(
    handler=dict(
        html_dict=dict(load="alphabet/fintabnet/structure_alphabet.txt"),
        cell_dict=dict(load="alphabet/fintabnet/character_alphabet.txt"),
    )
)

train_dataloader = dict(
    dataset=dict(
        ann_file="~/data/mutab_fintabnet.pkl",
        filter_cfg=dict(split="train"),
    ),
)

val_dataloader = dict(
    dataset=dict(
        ann_file="~/data/mutab_fintabnet.pkl",
        filter_cfg=dict(split="test"),
    ),
)

test_dataloader = dict(
    dataset=dict(
        ann_file="~/data/mutab_fintabnet.pkl",
        filter_cfg=dict(split="val"),
    ),
)
