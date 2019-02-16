mkdir output

cd static_model
#python dataset_feat_extractor.py --mode resnet50 --output_img

cd ../temporal_model
python test_temporal.py --dir ../output/static_resnet50 --model CLSTM_00_014999.pth --overlay

#python main.py --dir ../ststic_model/PATH/TO/YOUR/STATIC/FEATURES --model ./checkpoint/PATH/TO/YOUR/MODEL --seql 5 --gt /PATH/TO/YOUR/DATASET
