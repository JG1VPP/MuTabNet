_base_ = "pubtabnet.py"


model = dict(
    handler=dict(
        html_dict_file="alphabet/fintabnet/structure_alphabet.txt",
        cell_dict_file="alphabet/fintabnet/character_alphabet.txt",
    )
)

data = dict(
    train=dict(
        img_prefix="../data/fintabnet/img_tables/train/",
        ann_file="../data/mmocr_fintabnet/train/",
    ),
    val=dict(
        img_prefix="../data/fintabnet/img_tables/test/",
        ann_file="../data/mmocr_fintabsub/test/",
    ),
    test=dict(
        img_prefix="../data/fintabnet/img_tables/val/",
        ann_file="../data/mmocr_fintabsub/val/",
    ),
)
