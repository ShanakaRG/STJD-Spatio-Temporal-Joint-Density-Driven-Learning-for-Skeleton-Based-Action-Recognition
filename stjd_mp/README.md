# Spatio-Temporal Joint Density Driven Learning for Skeleton-Based Action Recognition
Shanaka Ramesh Gunasekara, Wanqing Li, Philip Ogunbona, Jack Yang

### Accepted by **TBIOM 2025**. [[Paper Link]](https://ieeexplore.ieee.org/abstract/document/10981864)

This repository includes Python (PyTorch) implementation of the STJD-MP.


# Abstract

Traditional approaches in unsupervised or self-supervised learning for skeleton-based action classification have concentrated predominantly on the dynamic aspects of skeletal sequences. Yet, the intricate interaction between the moving and static elements of the skeleton presents a rarely tapped discriminative potential for action classification. This paper introduces a novel measurement, referred to as spatial-temporal joint density (STJD), to quantify such interaction. Tracking the evolution of this density throughout an action can effectively identify a subset of discriminative moving and/or static joints termed prime joints to steer self-supervised learning. A new contrastive learning strategy named STJD-CL is proposed to align the representation of a skeleton sequence with that of its prime joints while simultaneously contrasting the representations of prime and non-prime joints. In addition, a method called STJD-MP is developed by integrating it with a reconstruction-based framework for more effective learning. Experimental evaluations on the NTU RGB+D 60, NTU RGB+D 120, and PKUMMD datasets in various downstream tasks demonstrate that the proposed STJD-CL and STJD-MP improved performance, particularly by 3.5 and 3.6 percentage points over the state-of-the-art contrastive methods on the NTU RGB+D 120 dataset using X-sub and X-set evaluations, respectively. The code is available at STJD.

# Requirements

```bash
python==3.8.13
torch==1.8.1+cu111
torchvision==0.9.1+cu111
tensorboard==2.9.0
timm==0.3.2
scikit-learn==1.1.1
tqdm==4.64.0
numpy==1.22.4
```

# Data Preparation

### Download datasets.
#### NTU RGB+D 60 and 120
1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract above files to `./data/nturgbd_raw`

#### PKU-MMD Phase I and Phase II
1. Request dataset here: http://39.96.165.147/Projects/PKUMMD/PKU-MMD.html
2. Download the skeleton data, label data, and the split files:
   1. `Skeleton.7z` + `Label_PKUMMD.7z` + `cross_subject.txt` + `cross_view.txt` (Phase I)
   2. `Skeleton_v2.7z` + `Label_PKUMMD_v2.7z` + `cross_subject_v2.txt` + `cross_view_v2.txt` (Phase II)
   3. Extract above files to `./data/pku_raw`

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
  - pku_v1/
  - pku_v2/
  - pku_raw/
    - v1/
      - label/
      - skeleton/
      - cross_subject.txt
      - cross_view.txt
    - v2/
      - label/
      - skeleton/
      - cross_subject_v2.txt
      - cross_view_v2.txt
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:
```
cd ./data/ntu # or cd ./data/ntu120
# Get skeleton of each performer
python get_raw_skes_data.py
# Remove the bad skeleton 
python get_raw_denoised_data.py
# Transform the skeleton to the center of the first frame
python seq_transformation.py
```
- Generate PKU-MMD Phase I or PKU-MMD Phase II dataset:
```
cd ./data/pku_v1 # or cd ./data/pku_v2
python pku_gendata.py
```

# Unsupervised Pre-Training

During the pre-training stage, we used the pre-trained weights from the MAMP encoder to initialize the model. Place them in ``` ./weights/ ``` The pre-training weights can be found  [[here.]](https://github.com/maoyunyao/MAMP?tab=readme-ov-file). Then refer the bashscrpts for the pre-training. 

# Fine-tuning

We provided scripts for four downstream tasks, Linear evalulation, fully supervised finetuning,  semi-supervised and transfer learning. 

Note that we are verifying the correctness of these scripts. If you find any problems with the code, please feel free to open an issue or contact us by sending an email to srg079[AT]uowmail.edu.au.


# Citation
If you find this work useful for your research, please consider citing our work:
```
@ARTICLE{10981864,
  author={Ramesh Gunasekara, Shanaka and Li, Wanqing and Ogunbona, Philip O. and Yang, Jie},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science}, 
  title={Spatio-Temporal Joint Density Driven Learning for Skeleton-Based Action Recognition}, 
  year={2025},
  volume={7},
  number={4},
  pages={632-642},
  keywords={Skeleton;Contrastive learning;Biometrics;Training;Kernel;Image reconstruction;Density measurement;Representation learning;Hands;Three-dimensional displays;Self-supervised learning;skeleton-based action recognition;spatio-temporal joint density},
  doi={10.1109/TBIOM.2025.3566212}}

```

# Acknowledgment
The framework of our code is based on [MAMP](https://github.com/maoyunyao/MAMP).
