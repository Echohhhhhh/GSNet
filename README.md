# GSNet
AAAI 2021. GSNet: Learning Spatial-Temporal Correlations from Geographical and Semantic Aspects for Traffic Accident Risk Forecasting


# Usage

train model on NYC:
```
python train.py --config config/nyc/GSNet_NYC_Config.json --gpus 0
```


train model on Chicago:
```
python train.py --config config/chicago/GSNet_Chicago_Config.json --gpus 0
```

# Configuration

The configuration file config.json contains three parts: Data, Training and Predict:


# About

If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{Wang2021gsnet,
  title={GSNet: Learning Spatial-Temporal Correlations from Geographical and Semantic Aspects for Traffic Accident Risk Forecasting},
  author={Beibei Wang, Youfang Lin,Shengnan Guo, Huaiyu Wan},
  booktitle={2021 AAAI Conference on Artificial Intelligence (AAAI'21)},
  year={2021} 
}
```