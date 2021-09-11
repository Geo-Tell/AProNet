# AProNet: Detecting Objects with Precise Orientation from Aerial Images
This is the repository of paper "AProNet: Detecting Objects with Precise Orientation from Aerial Images"
## Installation
### Requirements
```
Python: 3.6  
PyTorch: 1.2.0
CUDA: 9.2    
CUDNN: 7.6.2  
```
### Installation
a. Create a conda virtual environment and activate it. 
```
conda create --name AProNet python=3.6 -y  
conda activate AProNet  
```
b. Install PyTorch. 
```
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=9.2 -c pytorch
```
c. Clone the geovision-AProNet repository
```
git clone https://github.com/ZhWL123456/geovision-AProNet.git
workdir=./geovision-AProNet
```
d. Compile the `poly_nms`:    
```
cd ${workdir}/maskrcnn_benchmark/utils/poly_nms  
python setup.py build_ext --inplace  
```
d. Compile the `DOTA-devikit` dependency:   
```
sudo apt-get install swig  
cd ${workdir}/maskrcnn_benchmark/DOTA_devkit/polyiou  
swig -c++ -python csrc/polyiou.i  
python setup.py build_ext --inplace  
```
## Setting the datasets
a. Prepare your dataset as the status format.   
This project use the json annotation file with COCO format.
Make your directory layout like this:
```
${data-dir}
└── trainset
    ├── images
    │   ├── 1.png
    │   └── 2.png
    └── labelTxt
        ├── 1.txt
        └── 2.txt
```
A example of the \*.txt files ('1' means the object is difficult):
```
x1 y1 x2 y2 x3 y3 x4 y4 plane 0
x1 y1 x2 y2 x3 y3 x4 y4 harbor 1
```
Run the following Python snippet, and it will generate the json annotation file:
```python
from txt2json import collect_unaug_dataset, convert
img_dic = collect_unaug_dataset( os.path.join( "trainset", "labelTxt" ) )
convert( img_dic, "trainset",  os.path.join( "trainset", "train.json" ) )
```
b. Edit the file `maskrcnn_benchmark/config/paths_catalog.py` (from line7 to 17) to set the dir of datasets.  
```python
    DATA_DIR = "datasets" #need to change
    DATASETS = {
        "dota_trainval_cut": {
            "img_dir": "${dataset}/trainval_cut/images",             #need to change
            "ann_file": "${dataset}/trainval_cut/trainval_cut.json"  #need to change
        },
        "dota_test_cut": {
            "img_dir": "${dataset}/test_cut/images",                 #need to change
            "ann_file": "${dataset}/test_cut/test_cut.json"          #need to change
        },
    }
```
c. If your dataset is DOTA(options):  
For DOTA, you need to run the scripts  `XX` to split the original images into chip images (e.g., 1024*1024), and convert annotations to mmdet's format.
