custom_imports = dict(
    imports=[
        "mutab.data",
        "mutab.loss",
        "mutab.model",
        "mutab.phase",
        "mutab.score",
    ]
)

max_len_html = 800
max_len_cell = 8000

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

gca = ["GCA"]
gcb = dict(ratio=0.0625, heads=1)

model = dict(
    type="TableScanner",
    encoder=dict(
        type="TableEncoder",
        backbone=dict(
            type="TableResNet",
            dim=3,
            out=512,
            gcb1=dict(depth=1, **gcb),
            gcb2=dict(depth=2, **gcb, gca=gca),
            gcb3=dict(depth=5, **gcb, gca=gca),
            gcb4=dict(depth=3, **gcb, gca=gca),
        ),
        blocks=[],
        heads=8,
        d_model=512,
        dropout=0.2,
    ),
    decoder=dict(
        type="TableDecoder",
        html_decoder=dict(
            type="TableCellDecoder",
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
            type="TableCellDecoder",
            blocks=[
                dict(
                    att1=dict(type="WindowAttention"),
                    att2=dict(type="GlobalAttention"),
                ),
            ],
        ),
        html_fetcher=dict(
            type="TableCellFetcher",
            blocks=[
                dict(
                    att1=dict(type="GlobalAttention"),
                    att2=dict(type="GlobalAttention"),
                ),
            ],
        ),
        bbox_locator=dict(type="TableCellLocator"),
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
        dict(type="KLLoss", key="back", rev="html"),
        dict(type="BBLoss", key="bbox", cls="html"),
        dict(type="BBLoss", key="zone", cls="html"),
    ],
    cell_loss=[
        dict(type="CELoss", key="cell"),
    ],
    handler=dict(
        type="TableHandler",
        html_dict=dict(
            type="TableLexicon",
            load="alphabet/pubtabnet/structure_alphabet.txt",
        ),
        cell_dict=dict(
            type="TableLexicon",
            load="alphabet/pubtabnet/character_alphabet.txt",
        ),
        SOC="D",
        revisor=dict(
            type="TableRevisor",
            pipeline=[
                dict(type="ToHTML"),
                dict(
                    type="TableCombine",
                    SOC=["<td></td>", "<td"],
                    EOC=["<td></td>", "</td>"],
                ),
                dict(
                    type="TableReplace",
                    replace=eb_tokens,
                ),
                dict(
                    type="TableReplace",
                    replace={
                        r"<td[^>]*>(?=.*</thead>)": r"\g<0><b>",
                        r"</td>(?=.*</thead>)": r"</b></td>",
                    },
                ),
                dict(
                    type="TableReplace",
                    replace={
                        "<b></b>": "",
                        "<b><b>": "<b>",
                        "</b></b>": "</b>",
                    },
                ),
            ],
        ),
        outputs=[
            "html",
            "cell",
            "bbox",
            "full",
        ],
        targets=[
            "img_path",
            "ori_shape",
            "img_shape",
            "type",
            "html",
            "cell",
            "bbox",
            "full",
        ],
    ),
)

pipeline = [
    dict(type="FillBbox", cell=["<td></td>", "<td"]),
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=520, keep_ratio=True),
    dict(type="Pad", size=(520, 520)),
    dict(type="FormBbox"),
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
            "type",
            "html",
            "cell",
            "bbox",
        ],
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    sampler=dict(
        type="DefaultSampler",
        shuffle=True,
    ),
    dataset=dict(
        type="TableDataset",
        ann_file="~/data/mutab_pubtabnet.pkl",
        filter_cfg=dict(split="train"),
        pipeline=pipeline,
        test_mode=False,
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=0,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="TableDataset",
        ann_file="~/data/mutab_pubtabnet.pkl",
        filter_cfg=dict(split="val"),
        indices=range(24),
        pipeline=pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=2,
    num_workers=0,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="TableDataset",
        ann_file="~/data/mutab_pubtabnet.pkl",
        filter_cfg=dict(split="val"),
        pipeline=pipeline,
        test_mode=True,
    ),
)

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=30,
)

val_cfg = dict(type="ValLoop")

test_cfg = dict(type="TestLoop")

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.001,
    ),
    clip_grad=dict(
        max_norm=10,
        norm_type=2,
    ),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.2,
        begin=0,
        end=100,
        by_epoch=False,
    ),
    dict(
        type="MultiStepLR",
        milestones=[25],
        gamma=0.1,
        by_epoch=True,
    ),
]

val_evaluator = dict(
    type="TEDS",
    prefix="full",
    ignore=None,
)

test_evaluator = dict(
    type="TEDS",
    prefix="full",
    ignore=None,
)

launcher = "pytorch"
