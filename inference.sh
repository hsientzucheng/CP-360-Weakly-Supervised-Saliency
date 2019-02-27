mkdir output

cd static_model
python dataset_feat_extractor.py --mode resnet50 -om -of -oi

cd ../temporal_model
python test_temporal.py --dir ../output/static_resnet50 --model CLSTM_model_released.pth --overlay
