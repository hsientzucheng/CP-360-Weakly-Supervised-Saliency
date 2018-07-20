# CP-360-Weakly-Supervised-Saliency
This is the partial code for [Cube Padding for Weakly-Supervised Saliency Prediction in 360° Videos](http://aliensunmin.github.io/project/360saliency/), including ResNet-50 static feature extractor and ConvLSTM temporal model.

## Dataset 
To get Wild-360 dataset, check our [project website](http://aliensunmin.github.io/project/360saliency/).

## Requirements
- python==2.7
- PyTorch==0.3
- tqdm, skvideo, cv2, scipy, matplotlib

## Run
- First you can download our model [here](https://drive.google.com/file/d/1uOI4c9ojCU0pvUHN4cdf-JYUyWqkf-gm/view?usp=sharing) and put it into "temporal_model/checkpoint".
- Input the corresponding directories to "static_model/run.sh" and "temporal_model/run_eval.sh"
- After having the model and setting up the environment, you can run this code directly by
```
bash run.sh
```

## TODOs
This code will be updated later soon.
- [ ] temporal training code
- [ ] full-version of Equi-Cube projection and CubePadding

