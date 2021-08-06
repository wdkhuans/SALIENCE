# 1 SALIENCE

This repo is the official implementation for *An Unsupervised User Adaptation Model for Multiple  Wearable Sensors Based Human Activity Recognition*

## 1.1 The framework of SALIENCE
<div align=center>
<img src="https://user-images.githubusercontent.com/50646282/128143924-866f1552-c1ec-4f27-a4f2-9a405aca2287.jpg" width="400" height="400" alt="framework"/><br/>
<div align=left>
 
# 2 Prerequisites
- Python 3.6.12
- PyTorch 1.2.0
- math, sklearn, tensorboardX
 
# 3 Data Preparation
 ## 3.1 Download datasets
 - PAMAP2: https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring
 - OPPORTUNITY:  https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition

 ## 3.2 Data Processing
Put downloaded data into the following directory structure
 ```python
- data/
  - pamap/
   - Protocol/
      ... # raw data of PAMAP2(e.g,subject101.dat)
  - opp/
   - OpportunityUCIDataset/
     - dataset/
      ... # raw data of OPPORTUNITY(e.g,S1-ADL1.dat)
```
  ## 3.3 Generating Data

- Generate PAMAP2 dataset

```python
 cd data
 # pre-precess for PAMAP2
 python pre_process.py --dataset pamap
```
 
- Generate OPPORTUNITY dataset

```python
 cd data
 # pre-precess for OPPORTUNITY
 python pre_process.py --dataset opp
```
 
 
 
# 4 Running

## 4.1 Training & Testing

- Change the config depending on what you want
```python
cd ..
# run on the PAMAP2
python main.py --dataset pamap
# run on the OPPORTUNITY
python main.py --dataset opp
# change the learning rate
python main.py --lr 0.0001
# change the batch size
python main.py --batch_size 64
```

# 5 Acknowledgements
This repo is based on [MCD_DA](https://github.com/mil-tokyo/MCD_DA). Great thanks to the original authors for their work!
 
 
# 6 Citation

Please cite this work if you find it useful.

If you have any question, feel free to contact: `shmiao@zju.edu.cn`

