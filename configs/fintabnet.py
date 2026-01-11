_base_ = "pubtabnet.py"


model = dict(
    handler=dict(
        html_dict=dict(load="alphabet/fintabnet/structure_alphabet.txt"),
        cell_dict=dict(load="alphabet/fintabnet/character_alphabet.txt"),
    )
)

pipeline = [
    dict(type="FillBbox", cell=["<td></td>", "<td"]),
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=520, keep_ratio=True),
    dict(type="Pad", size=(520, 520)),
    dict(type="FillBbox"),
    dict(type="FormBbox"),
    dict(type="FlipBbox"),
    dict(type="Hardness"),
    dict(type="ToOTSL"),
    dict(
        type="Normalize",
        mean=[128, 128, 128],
        std=[128, 128, 128],
    ),
    dict(type="ImageToTensor", keys=["img"]),
    dict(
        type="Annotate",
        keys=["img"],
        meta=[
            "img_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "html",
            "cell",
            "bbox",
            "type",
        ],
    ),
]

train_dataloader = dict(
    dataset=dict(
        ann_file="~/data/mutab_fintabnet.pkl",
        filter_cfg=dict(split="train"),
        pipeline=pipeline,
    ),
)

val_dataloader = dict(
    dataset=dict(
        ann_file="~/data/mutab_fintabnet.pkl",
        filter_cfg=dict(split="test"),
        pipeline=pipeline,
    ),
)

test_dataloader = dict(
    dataset=dict(
        ann_file="~/data/mutab_fintabnet.pkl",
        filter_cfg=dict(split="val"),
        pipeline=pipeline,
    ),
)
