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
- Other dependencies:
    - tqdm, scipy, matplotlib, PIL, ruamel_yaml, collections

## Model
### CubePadding

<img src="./images/cubeprojection.jpg" width="50%"/>

<img src="./images/feature_map_visual.jpg" width="50%"/>

You can test the cube padding module by simply execute cube_pad.py
```
python [CP-360-Weakly-Supervised-Saliency PATH]/model/cube_pad.py
```

### Pretrained model
You can download our convolution LSTM model [here](https://drive.google.com/file/d/1uOI4c9ojCU0pvUHN4cdf-JYUyWqkf-gm/view?usp=sharing)
The model should be put into the directory:
```
[CP-360-Weakly-Supervised-Saliency PATH]/checkpoint/CLSTM_model_released.pth
```

## Inference
- To run the inference process, you should first modify the config file
```
vim [CP-360-Weakly-Supervised-Saliency PATH]/config.yaml
```

- After installing requirements and setting up the configurations, the static model can be run as:
```
cd static_model
python dataset_feat_extractor.py --mode resnet50 --output_img
```

- Having the features from the static model, run the temporal model by:
```
cd temporal_model
python test_temporal.py --dir ../output/static_resnet50 --model CLSTM_model_released.pth --overlay
```

- These commands are in the script, just run:
```
bash inference.sh
```

## Results
<img src="./images/result.jpg" width="80%"/>
In each block, consecutive frames of various methods, ground truth, and raw videos are shown in the left panel. We highlight regions for comparison using white dash rectangles. In the right panel, one example is zoom-in (red box) and two salient NFoVs (yellow boxes) are rendered. Our temporal method (Ours) significantly outperforms others in overcoming distortion, image boundaries, and smoothness in time. See more examples in supplementary materials.

