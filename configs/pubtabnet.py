max_len_html = 800
max_len_cell = 8000

seed = None

eb_tokens = {
    "<eb></eb>": "<td></td>",
    "<eb1></eb1>": "<td> </td>",
    "<eb2></eb2>": "<td><b> </b></td>",
    "<eb3></eb3>": "<td>\u2028\u2028</td>",
    "<eb4></eb4>": "<td><sup> </sup></td>",
    "<eb5></eb5>": "<td><b></b></td>",
    "<eb6></eb6>": "<td><i> </i></td>",
    "<eb7></eb7>": "<td><b><i></i></b></td>",
    "<eb8></eb8>": "<td><b><i> </i></b></td>",
    "<eb9></eb9>": "<td><i></i></td>",
    "<eb10></eb10>": "<td><b> \u2028 \u2028 </b></td>",
}

revisions = {
    "^.*$": eb_tokens,
    "<thead>(.*?)</thead>": {
        r'(<td( [a-z]+="(\d)+")*?>)(.*?)</td>': r"\g<1><b>\g<4></b></td>",
        "<b></b>": "",
        "<b><b>": "<b>",
        "</b></b>": "</b>",
    },
}

cell_tokens = ["<td></td>", "<td", *eb_tokens]

gca = ["GCA"]
gcb = dict(ratio=0.0625, heads=1)

model = dict(
    type="TableScanner",
    backbone=dict(
        type="TableResNet",
        dim=3,
        out=512,
        gcb1=dict(depth=1, **gcb),
        gcb2=dict(depth=2, **gcb, gca=gca),
        gcb3=dict(depth=5, **gcb, gca=gca),
        gcb4=dict(depth=3, **gcb, gca=gca),
    ),
    encoder=dict(
        type="PositionalEncoding2D",
        channels=512,
    ),
    decoder=dict(
        type="TableDecoder",
        html_decoder=dict(
            blocks=[
                dict(
                    att1=dict(type="WindowAttention"),
                    att2=dict(type="GlobalAttention"),
                ),
                dict(
                    att1=dict(type="WindowAttention"),
                    att2=dict(type="GlobalAttention"),
                ),
                dict(
                    att1=dict(type="WindowAttention"),
                    att2=dict(type="GlobalAttention"),
                ),
            ],
        ),
        cell_decoder=dict(
            blocks=[
                dict(
                    att1=dict(type="WindowAttention"),
                    att2=dict(type="GlobalAttention"),
                ),
            ],
        ),
        html_fetcher=dict(
            blocks=[
                dict(
                    att1=dict(type="GlobalAttention"),
                    att2=dict(type="GlobalAttention"),
                ),
            ],
        ),
        bbox_locator=dict(pass_html=True),
        heads=8,
        window=300,
        d_model=512,
        dropout=0.2,
        max_len_html=max_len_html,
        max_len_cell=max_len_cell,
    ),
    html_loss=[
        dict(type="CELoss", key="html"),
        dict(type="CELoss", key="back"),
        dict(type="KLLoss", key="html", rev="back"),
        dict(type="BBLoss"),
    ],
    cell_loss=[
        dict(type="CELoss", key="cell"),
    ],
    handler=dict(
        type="TableHandler",
        html_dict_file="alphabet/pubtabnet/structure_alphabet.txt",
        cell_dict_file="alphabet/pubtabnet/character_alphabet.txt",
        SOC=["<td></td>", "<td"],
        EOC=["<td></td>", "</td>"],
        revisor=dict(
            template="{}",
            patterns=revisions,
        ),
    ),
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="TableResize", size=520),
    dict(
        type="TablePad",
        size=(520, 520),
    ),
    dict(type="TableBboxEncode"),
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
            "html",
            "cell",
            "bbox",
        ],
    ),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="TableResize", size=520),
    dict(
        type="TablePad",
        size=(520, 520),
    ),
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
        ],
    ),
]

loader = dict(
    type="TableHardDiskLoader",
    max_len_html=max_len_html,
    parser=dict(
        type="TableStrParser",
        cell_tokens=cell_tokens,
    ),
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type="TableDataset",
        img_prefix="../data/pubtabnet/train/",
        ann_file="../data/mmocr_pubtabnet/train/",
        pipeline=train_pipeline,
        loader=loader,
        test_mode=False,
    ),
    val=dict(
        type="TableDataset",
        img_prefix="../data/pubtabnet/val/",
        ann_file="../data/mmocr_pubtabsub/val/",
        pipeline=train_pipeline,
        loader=loader,
        test_mode=True,
    ),
    test=dict(
        type="TableDataset",
        img_prefix="../data/pubtabnet/val/",
        ann_file="../data/mmocr_pubtabsub/val/",
        pipeline=test_pipeline,
        loader=loader,
        test_mode=True,
    ),
)

# optimizer
optimizer = dict(type="Ranger", lr=1e-3)
optimizer_config = dict(grad_clip=dict(max_norm=30, norm_type=2))

# learning policy
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=50,
    warmup_ratio=1.0 / 3,
    step=[25, 28],
)

# runner
runner = dict(type="EpochBasedRunner", max_epochs=30)

# evaluation
ignore = None
evaluation = dict(interval=1, metric="acc")

# fp16
fp16 = dict(loss_scale="dynamic")

# checkpoint setting
checkpoint_config = dict(interval=1)

# log_config
log_config = dict(interval=100, hooks=[dict(type="TextLoggerHook")])

# logger
log_level = "INFO"

# yapf:enable
dist_params = dict(backend="nccl")

# pretrained
load_from = None
resume_from = None

# workflow
workflow = [("train", 1)]
