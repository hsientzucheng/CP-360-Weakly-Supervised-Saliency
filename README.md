# CP-360-Weakly-Supervised-Saliency
This is the code for [Cube Padding for Weakly-Supervised Saliency Prediction in 360° Videos](http://aliensunmin.github.io/project/360saliency/), including ResNet-50 static feature extractor and ConvLSTM temporal model.

## Getting Started
Clone the repo:
```
git clone https://github.com/hsientzucheng/CP-360-Weakly-Supervised-Saliency.git
```

## Dataset 
To get Wild-360 dataset, check our [project website](http://aliensunmin.github.io/project/360saliency/).

## Requirements
Tested under
- Python == 3.6
- PyTorch >= 0.3
- cv2 == 3.4.2
- tqdm, scipy, matplotlib, PIL, ruamel_yaml, collections

## Model
### CubePadding
You can test the cube padding module by simply execute cube_pad.py
```
python [PATH of CP-360-Weakly-Supervised-Saliency]/model/cube_pad.py
```
### Pretrained model
You can download our convolution LSTM model [here](https://drive.google.com/file/d/1uOI4c9ojCU0pvUHN4cdf-JYUyWqkf-gm/view?usp=sharing)
The model should be put into the directory:
```
[PATH of CP-360-Weakly-Supervised-Saliency]/checkpoint/model.pth
```

## Inference
- To run the inference process, you should first modify the config file
- After having the model, installing requirements, and setting up the configurations

```
bash run.sh
```

## TODOs
[v] visualization code
[v] revised cube padding
[] clstm training code
