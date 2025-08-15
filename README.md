# SeCoV2 (TPAMI-2025)

SeCoV2: Semantic-Connectivity-driven-Pseudo-Labeling-for-Robust-Cross-Domain-Semantic-Segmentation 

This is an extended journal version of the previous Pseudo-label denoising work SeCo, named SeCoV2, which introduces an uncertainty-aware correction module that constructs a connectivity graph and enforces relational consistency for robust refinement in ambiguous regions. 

## :gear: Pipeline
![](./images/Fig3_Pipeline.png)

SeCoV2 also broadens the applicability of SeCo by extending evaluation to more challenging scenarios:

![](./images/Fig2_Motivation.png)


![](./images/Fig4_extend.png)

## :card_index_dividers: Data

We are currently organizing the open-set data synthesized by the diffusion model used in OSDA, and plan to release it publicly.



## :jigsaw: Requirements

```
Python 3.8.0
pytorch 1.10.1
torchvision 0.11.2
einops  0.3.2
```
Please see `requirements.txt` for all the other requirements.

You can use PSA and SCC to obtain high-purity connectivity-based pseudo-labels. 
These pseudo-labels can then be exploited and embedded into existing unsupervised domain adaptative semantic segmentation methods.

## :hammer_and_wrench: Pixel Semantic Aggregation

First, you can obtain pixel-level pseudo-labels by pixel thresholding (e.g.[cbst](https://github.com/yzou2/cbst) ) from a UDA method (e.g. [ProDA](https://github.com/microsoft/ProDA) ) or a source-free UDA method (e.g. [DTST](https://github.com/DZhaoXd/DT-ST)), or a UDG method (e.g. [SHADE](https://github.com/HeliosZhao/SHADE) ).  
And organize them in the following format.   
```
"""
├─image
├─pixel-level pseudo-label
└─list
"""
list (XXX.txt) records the image names (XXX.png) and their corresponding pixel-level pseudo-labels.
```
Then, run the PSA as follows:
```
${exp_name}="HRDA_seco"
CUDA_VISIBLE_DEVICES="1"  nohup python seco_psa_v2.py --id-list-path  ./splits/cityscapes/${exp_name}/all.txt --class-num ${class_name}  > logs/${exp_name} 2>&1 &
```
Afterward, you can find the aggregated pseudo-labels and its records in `root_path/${exp_name}_vit_{B/H}`.

## :hammer_and_wrench: Semantic Connectivity Correction
After PSA, the noise is also amplified, and then you can use SCC to denoise the connected regions. 
Refer to ([SCC_v1](https://github.com/DZhaoXd/SeCoV2/tree/main/SCC_V1) ) or ([SCC_v2](https://github.com/DZhaoXd/SeCoV2/tree/main/SCC_V2) ) part for specific instructions.


## :speech_balloon: License
Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.

## Acknowledgement
Many thanks to those wonderful work and the open-source code.
- [Segment Anything](https://segment-anything.com/) 
- [BETA](https://github.com/xyupeng/BETA)

