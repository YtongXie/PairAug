# PairAug

<p align="left">
    <img src="overall.png" width="85%" height="85%">
</p>

This repo holds the Pytorch implementation of PairAug:<br />

**[CVPR2024] PairAug: What Can Augmented Image-Text Pairs Do for Radiology?** 


## Usage
* Create a new conda environment 
```
conda create --name pairaug python=3.10
source activate pairaug
```
* Clone this repo
```
git clone https://github.com/YtongXie/PairAug.git
cd PairAug
```
### Medical image generation
#### 0. Installation 
* Install packages for image generation
```
pip install -r requirements_T2I.txt
```
#### 1. Data Preparation
* Download [MIMIC-CXR-JPG dataset](https://physionet.org/content/mimic-cxr-jpg/2.1.0/)(Need to be a credentialed user for downloading) <br/>

* Put the image data under `data/MIMIC_images_ori/` should be like:
``` data/MIMIC_images_ori/
    ├── p19_p19995997_s50123635_6fa953ea-79c237a5-4ca3be78-e3ae6427-c327e17b.png
    ├── p19_p19996061_s58482960_87923de8-5595ad44-eaa89d38-610e97e2-42cacf04.png
    ├── p19_p19996762_s58960501_d1aa8bb2-afa746e5-7ff2d875-045be82e-9da2236e.png
    ├── p19_p19996786_s52281280_ff766df0-156bec0b-d33db351-26d340dd-4711ae6f.png
    ├── p19_p19997087_s56819576_d9eedfc8-38f766ac-309f0c5f-cea553c3-d06b9b23.png
    ├── p19_p19997293_s53859051_b7c6e487-b22d02c1-577476ea-07fa9244-9aed3945.png
    ├── ...
```
* Put the report data under `data/MIMIC_reports_ori/` should be like:
``` data/MIMIC_reports_ori/
    ├── p19_p19995997_s50123635_6fa953ea-79c237a5-4ca3be78-e3ae6427-c327e17b.txt
    ├── p19_p19996061_s58482960_87923de8-5595ad44-eaa89d38-610e97e2-42cacf04.txt
    ├── p19_p19996762_s58960501_d1aa8bb2-afa746e5-7ff2d875-045be82e-9da2236e.txt
    ├── p19_p19996786_s52281280_ff766df0-156bec0b-d33db351-26d340dd-4711ae6f.txt
    ├── p19_p19997087_s56819576_d9eedfc8-38f766ac-309f0c5f-cea553c3-d06b9b23.txt
    ├── p19_p19997293_s53859051_b7c6e487-b22d02c1-577476ea-07fa9244-9aed3945.txt
    ├── ...
```
* Run `python data/reports_jsonl.py` to generate the report data lists `data/metadata_snets.jsonl`.
