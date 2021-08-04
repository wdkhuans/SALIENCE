# SALIENCE

This repo is the official implementation for *An Unsupervised User Adaptation Model for Multiple  Wearable Sensors Based Human Activity Recognition*

## The framework of SALIENCE
<div align=center>
<img src="https://user-images.githubusercontent.com/50646282/128143924-866f1552-c1ec-4f27-a4f2-9a405aca2287.jpg" width="600" height="600" alt="framework"/><br/>
<div align=left>
 
# Prerequisites
- Python == 3.6.12
- PyTorch == 1.2.0
- math, sklearn, tensorboardX
 
 # Data Preparation
 ### Download datasets
 - PAMAP2: https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring
 - OPPORTUNITY:  https://archive.ics.uci.edu/ml/datasets/OPPORTUNITY+Activity+Recognition

 ### Data Processing
Put downloaded data into the following directory structure:
 ```python
- data/
  - Protocol/
      ... # raw data of PAMAP2(e.g,subject101.dat)
```
 ### Generating Data

- Generate PAMAP2 dataset:

```python
 cd data
 # pre-precess for PAMAP2
 python pre_process.py
```
# Training & Testing

### Training

- Change the config depending on what you want.
```python
cd ..
python main.py --lr 0.0001
```
 
# Citation

Please cite this work if you find it useful.


# Contact
For any questions, feel free to contact: `shmiao@zju.edu.cn`
 
 
