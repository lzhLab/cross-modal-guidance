# Cross-Modal Guidance for Liver Lesion segmentation on Multi-Phase CT Images

  This repository contains the source code for our paper


## Abstract

  Liver lesion segmentation plays an important role in the computer-aided diagnosis system of liver. The segmentation accuracy of the automatic liver lesion segmentation system can be improved by utilizing multi-phase CT images. It is usually necessary to go through the registration step to align multimodal images, but the segmentation model cannot efficiently utilize the information when the registration results are not satisfactory enough. In this paper, we propose a new network with the Cross-Multimodal Guidance (CMG) module to try to solve this problem. To effectively utilize multi-phase CT images, we devise a Multi-Scale Fusion module to extract supplementary information useful for liver lesion segmentation from each phase. This network allows to obtain cross-modal guidance maps to mutually guide the segmentation between poorly registered modalities, leading to a more efficient use of multimodal images. We conduct extensive experiments on the MPCT dataset of liver and liver tumor and on the multimodal dataset from the CHAOS challenge. Experimental results demonstrate that our method outperforms existing SOTA methods in the tumor segmentation of multi-phase CT images, and also show that our method has comparable performance in other multimodal image segmentation tasks.



## Usage

You can run the main program `main.py` in the python environment. Our proposed network is implemented using the open source pytorch framework. The required python dependency package versions are as follows:
```
python = 3.8
pytorch = 1.8
numpy = 1.19
scikit-learn = 1.1.3
```


## Datasets

We trained on a private MPTH dataset and a public CHAOS dataset. The data directory should be organized as follows:

```
├── datasets
|   ├── data
|   	├── MPTH
|   		├── train_img
|   			├── pv
|   				├── 1
|   				    ├── 1.png
|   				    ├── 2.png
|   				    ......
|   				├── 2
|   				......
|   			├── art
|   				├── 1
|   				    ├── 1.png
|   				    ├── 2.png
|   				    ......
|   				├── 2
|   				......
|   		├── liver_mask
|   			├── pv
|   			├── art
|   		├── tumor_mask
|   			├── pv
|   			├── art
|   	├── CHAOS
		......
```
