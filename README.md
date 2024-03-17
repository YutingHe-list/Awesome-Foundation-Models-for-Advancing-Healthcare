# Awesome-Foundation-Models-for-Advancing-Healthcare

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

This repo is a collection of AWESOME things about Foundation models in healthcare, including language foundation models (LFMs), vision foundation models (VFMs), bioinformatics foundation models (BFMs), and multimodal foundation models (MFMs). Feel free to star and fork.

# Contents
- [Awesome-Foundation-Models-for-Advancing-Healthcare](#Awesome-Foundation-Models-for-Advancing-Healthcare)
- [Related Survery](#survey)
- [Methods](#methods)
  - [LFMs](#lfm)
    - [LFM pre-training](#pre_training_lfm)
    - [LFM adaptation](#adaptation_lfm)
  - [VFMs](#vfm)
    - [VFM pre-training](#pre_training_vfm)
    - [VFM adaptation](#adaptation_vfm)
  - [BFMs](#bfm)
    - [BFM pre-training](#pre_training_bfm)
    - [BFM adaptation](#adaptation_bfm)
  - [MFMs](#mfm)
    - [MFM pre-training](#pre_training_mfm)
    - [MFM adaptation](#adaptation_mfm)
  - [Benchmarks](#benchmarks)
- [Datasets](#datasets)
- [Lectures and Tutorials](#lectures-and-tutorials)
- [Other Resources](#other-resources)

# Papers
## Survery


## Open set image domain adaptation

### ...

**Arxiv**

**Coference**

***2017***

[ICCV][Open Set Domain Adaptation](https://openaccess.thecvf.com/content_iccv_2017/html/Busto_Open_Set_Domain_ICCV_2017_paper.html)

***2018***

[ECCV][Open Set Domain Adaptation by Backpropagation](https://openaccess.thecvf.com/content_ECCV_2018/html/Kuniaki_Saito_Adversarial_Open_Set_ECCV_2018_paper.html)

***2019***

[ICIP][Improved Open Set Domain Adaptation with Backpropagation](https://ieeexplore.ieee.org/abstract/document/8803287)

[CVPR][Separate to Adapt: Open Set Domain Adaptation via Progressive Separation](https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_Separate_to_Adapt_Open_Set_Domain_Adaptation_via_Progressive_Separation_CVPR_2019_paper.html)

[CVPR][Weakly Supervised Open-Set Domain Adaptation by Dual-Domain Collaboration](https://openaccess.thecvf.com/content_CVPR_2019/html/Tan_Weakly_Supervised_Open-Set_Domain_Adaptation_by_Dual-Domain_Collaboration_CVPR_2019_paper.html)

***2020***

[ECCV][Multi-source Open-Set Deep Adversarial Domain Adaptation](https://linkspringer.53yu.com/chapter/10.1007/978-3-030-58574-7_44)

[ECCV][On the Effectiveness of Image Rotation for Open Set Domain Adaptation](https://linkspringer.53yu.com/chapter/10.1007/978-3-030-58517-4_25)

[CVPR][Exploring Category-Agnostic Clusters for Open-Set Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2020/html/Pan_Exploring_Category-Agnostic_Clusters_for_Open-Set_Domain_Adaptation_CVPR_2020_paper.html)

[ICML][Progressive Graph Learning for Open-Set Domain Adaptation](https://proceedings.mlr.press/v119/luo20b.html)

[CVPR][Towards Inheritable Models for Open-Set Domain Adaptation](https://openaccess.thecvf.com/content_CVPR_2020/html/Kundu_Towards_Inheritable_Models_for_Open-Set_Domain_Adaptation_CVPR_2020_paper.html)

***2021***

[ICCV][Towards Novel Target Discovery Through Open-Set Domain Adaptation](https://openaccess.thecvf.com/content/ICCV2021/html/Jing_Towards_Novel_Target_Discovery_Through_Open-Set_Domain_Adaptation_ICCV_2021_paper.html)

[WACV][Object Recognition With Continual Open Set Domain Adaptation for Home Robot](https://openaccess.thecvf.com/content/WACV2021/html/Kishida_Object_Recognition_With_Continual_Open_Set_Domain_Adaptation_for_Home_WACV_2021_paper.html)

***2022***

[WACV][Distance-Based Hyperspherical Classification for Multi-Source Open-Set Domain Adaptation](https://openaccess.thecvf.com/content/WACV2022/html/Bucci_Distance-Based_Hyperspherical_Classification_for_Multi-Source_Open-Set_Domain_Adaptation_WACV_2022_paper.html)

**Journal**

***2020***

[TPAML][Open Set Domain Adaptation for Image and Action Recognition](https://ieeexplore.ieee.org/abstract/document/8531764/)

***2021***

[TNNLS][Open Set Domain Adaptation: Theoretical Bound and Algorithm](https://s3-us-west-2.amazonaws.com/ieeeshutpages/xplore/xplore-shut-page.html)

[AAAI][Balanced Open Set Domain Adaptation via Centroid Alignment](https://ojs.aaai.org/index.php/AAAI/article/view/16977)

[TNNLS][Bridging the Theoretical Bound and Deep Algorithms for Open Set Domain Adaptation](https://ieeexplore.ieee.org/abstract/document/9594518)

***2022***

[Measurement][Feature distance-based deep prototype network for few-shot fault diagnosis under open-set domain adaptation scenario](https://www.sciencedirect.com/science/article/pii/S0263224122007448)

## Open set medical image analysis

### ...

**Arxiv**

[2022][Test Time Transform Prediction for Open Set Histopathological Image Recognition](https://arxiv.org/abs/2206.10033) [code](https://github.com/agaldran/t3po)

[2022][Open-Set Recognition of Breast Cancer Treatments](https://arxiv.org/abs/2201.02923)

[2019][Open Set Medical Diagnosis](https://arxiv.org/abs/1910.02830)

**Coference**

[MICCAI 2022][Delving into Local Features for Open-Set Domain Adaptation in Fundus Image Analysis](...)

[MICCAI 2018][Evaluation of Various Open-Set Medical Imaging Tasks with Deep Neural Networks](https://arxiv.org/abs/2110.10888)

[ACM 2022][SODA: Detecting Covid-19 in Chest X-rays with Semi-supervised Open Set Domain Adaptation](https://arxiv.org/abs/2005.11003)

## Open set in segmentation

### ...

**Arxiv**

[2022][Conditional Reconstruction for Open-set Semantic Segmentation](https://arxiv.org/abs/2203.01368)

[2020][Fully Convolutional Open Set Segmentation](https://arxiv.org/abs/2006.14673)

[2020][Towards Open-Set Semantic Segmentation of Aerial Images](https://arxiv.org/abs/2001.10063)

**Coference**

[CVPR 2022][SimT: Handling Open-set Noise for Domain Adaptive Semantic Segmentation](https://arxiv.org/abs/2203.15202)

[ECCV 2018][Bayesian Semantic Instance Segmentation in Open Set World](https://arxiv.org/abs/1806.00911)

## Open set in anomaly detection

### ...

**Arxiv**

[2022][The Familiarity Hypothesis: Explaining the Behavior of Deep Open Set Methods](https://arxiv.org/abs/2203.02486)

[2021][Deep Compact Polyhedral Conic Classifier for Open and Closed Set Recognition](https://arxiv.org/abs/2102.12570)

**Coference**

[ECCV 2022][DenseHybrid: Hybrid Anomaly Detection for Dense Open-set Recognition](https://arxiv.org/abs/2207.02606)

[CVPR 2022][Catching Both Gray and Black Swans: Open-set Supervised Anomaly Detection](https://arxiv.org/abs/2203.14506)

[CVPR 2022][UBnormal: New Benchmark for Supervised Open-Set Video Anomaly Detection](https://arxiv.org/abs/2111.08644)

**Journal**


## Benchmarks

# Library

# Lectures and Tutorials

# Other Resources

|                           Dataset                            | Number | Modality  |     Region     |     Format      |
| :----------------------------------------------------------: | :----: | :-------: | :------------: | :-------------: |
|                                                              |   10   |  4D  CT   |      Lung      |      .img       |



