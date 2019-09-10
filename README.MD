# Efficient and Accurate Arbitrary-Shaped Text Detection with Pixel Aggregation Network

![](imgs/paper/PAN.jpg)

## Requirements
* pytorch 1.1
* torchvision 0.3
* pyclipper
* opencv3


## Data Preparation

prepare a text in the following format, use '\t' as a separator
```bash
/path/to/img.jpg path/to/label.txt
...
```

## Train
1. config the `train_data_path`,`val_data_path`in [config.json](config.json)
2. use following script to run
```sh
python3 train.py
```

## Test

[eval.py](eval.py) is used to test model on test dataset

1. config `model_path`, `img_path`, `gt_path`, `save_path` in [eval.py](eval.py)
2. use following script to test
```sh
python3 eval.py
```

## Predict 
[predict.py](predict.py) is used to inference on single image

1. config `model_path`, `img_path`, in [predict.py](predict.py)
2. use following script to predict
```sh
python3 predict.py
```

The project is still under development.

<h2 id="Performance">Performance</h2>

| Method                   | Precision (%) | Recall (%) | F-measure (%) | FPS(1080Ti) |
|--------------------------|---------------|------------|---------------|-----|
| paper(short_size:736)  | 84.0 | 81.9 | 82.9 | 26.1 |
| my implementation(short_size:736)  | TBD | TBD | TBD | TBD |

### [ICDAR 2015](http://rrc.cvc.uab.es/?ch=4)

TBD

### examples
TBD

### reference
1. https://arxiv.org/pdf/1908.05900.pdf
2. https://github.com/WenmuZhou/PSENet.pytorch
