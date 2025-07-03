_base_ = "pubtabnet.py"


model = dict(
    handler=dict(
        html_dict_file="alphabet/fintabnet/structure_alphabet.txt",
        cell_dict_file="alphabet/fintabnet/character_alphabet.txt",
    )
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="TableResize", size=520),
    dict(
        type="TablePad",
        size=(520, 520),
    ),
    dict(type="TableBboxFlip"),
    dict(type="TableBboxEncode"),
    dict(type="ToOTSL"),
    dict(type="ToTensorOCR"),
    dict(
        type="NormalizeOCR",
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5],
    ),
    dict(
        type="Collect",
        keys=["img"],
        meta_keys=[
            "filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "img_scale",
            "rows",
            "cols",
            "html",
            "cell",
            "bbox",
        ],
    ),
]

data = dict(
    train=dict(
        img_prefix="../data/fintabnet/img_tables/train/",
        ann_file="../data/mmocr_fintabnet/train/",
        pipeline=train_pipeline,
    ),
    val=dict(
        img_prefix="../data/fintabnet/img_tables/test/",
        ann_file="../data/mmocr_fintabsub/test/",
        pipeline=train_pipeline,
    ),
    test=dict(
        img_prefix="../data/fintabnet/img_tables/val/",
        ann_file="../data/mmocr_fintabsub/val/",
    ),
)
