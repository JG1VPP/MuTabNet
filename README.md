# MuTabNet

End-to-End table OCR model using a hierarchical Transformer that outputs HTML tags and cell contents.

## Usage

### Install

Install [PyTorch](https://pytorch.org) 2.0 and run the following command:

```sh
pip install -e .
```

### Models

See [releases](https://github.com/JG1VPP/MuTabNet/releases).

### Datasets

Download the following datasets:

- [FinTabNet](https://developer.ibm.com/data/fintabnet)
- [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet)
- [ICDAR Task-B Test Data](https://github.com/ajjimeno/icdar-task-b)

### Preprocess

Run [preprocess.py](preprocess.py) as follows:

```sh
$ python preprocess.py datasets/FinTabNet.yaml
$ python preprocess.py datasets/PubTab250.yaml
$ python preprocess.py datasets/PubTabNet.yaml
```

The datasets must be placed in `data` directory as follows:

```sh
$ ls ~/data
fintabnet/
  img_tables/
    train/
      100000_61623.png
      100001_61624.png
      100002_61625.png
      100003_61626.png
      100004_61627.png
    val/
icdar-task-b/
  final_eval/
    000221630ba33f9118f2671a715d6962e08d6b76a5a0c77a9fe26c291df763b0.png
    0005e8fe1b3ba14982336837219f285921af7c152cfc81ac88bcf52809299279.png
    002b1bf2bbb7dd7ec6201174e68df6346f448cd3951e861c3f940711c769f25f.png
    002bfeebe20be2e97fab46b99ce68321afb8972f6d8f131f0c1f5392819d3a23.png
    002c7215e95cd4bfebffb13dc0db32ab229a6674f4f1add84518ae52b75ac0da.png
  final_eval.json
mutab_fintabnet.pkl
mutab_pubtab250.pkl
mutab_pubtabnet.pkl
pubtabnet/
  PubTabNet_2.0.0.jsonl
  train/
    PMC1064074_007_00.png
    PMC1064076_003_00.png
    PMC1064076_004_00.png
    PMC1064080_002_00.png
    PMC1064094_007_00.png
  val/
```

### Training

Run [train.py](train.py) to start training using four GPUs:

```sh
name=pubtab250
save=~/work/$name

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29500 ./train.sh ./configs/$name.py $save 4
```

### Evaluation

Run [test.py](test.py) to evaluate the model and calculate TEDS score:

```sh
path=~/data/icdar-task-b/final_eval
json=~/data/icdar-task-b/final_eval.json

python test.py --conf ./configs/$name.py --ckpt $save/latest.pth --path $path --json $json
```

For FinTabNet, we use validation set including 10,656 tables as test set in imitation of the previous work.

## Requirements

We recommend that you use at least four NVIDIA V100 32GB GPUs.

## License

This project is licensed under the MIT License.
See [LICENSE](LICENSE) for more details.

## Citation

```latex
@inproceedings{ICDAR24KAT,
  author={Takaya Kawakatsu},
  title={Multi-Cell Decoder and Mutual Learning for Table Structure and Character Recognition},
  booktitle={Document Analysis and Recognition - ICDAR 2024},
  publisher={Springer Nature Switzerland},
  year={2024},
  pages={389--405},
}
```
