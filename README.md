# GRCOD
## Towards Efficient and Robust Generative Models for Referring Camouflaged Object Detection
![Static Badge](https://img.shields.io/badge/Apache-blue?style=flat&label=license&labelColor=black&color=blue)
![Static Badge](https://img.shields.io/badge/passing-green?style=flat&label=build&labelColor=black&color=green)
![Static Badge](https://img.shields.io/badge/passing-green?style=flat&label=circleci&labelColor=black&color=green)
![Static Badge](https://img.shields.io/badge/welcome-green?style=flat&label=PRs&labelColor=black&color=green)
![Static Badge](https://img.shields.io/badge/Python-green?style=flat&label=Language&labelColor=black&color=green)

## 🎮 [Appendix](https://github.com/QuantumScriptHub/GRCOD/blob/main/data/ICME2025_Appendix.pdf)
Since our GRCOD is the first generative approach for Ref-COD, there are no existing generative methods available for direct comparison. Therefore, following the standard idea of fine-tuning Stable Diffusion, we design another generative baseline for reference, in order to better highlight the efficiency and robustness of GRCOD. Please refer to the [supplementary material](https://github.com/QuantumScriptHub/GRCOD/blob/main/data/ICME2025_Appendix.pdf) for details. 

In our supplementary materials, you will find a more detailed theoretical analysis of generative Ref-COD approach, along with comprehensive experimental details and comparisons. Make sure not to miss it!

##  📢 Overview
<p align="justify">
Existing Referring Camouflaged Object Detection (Ref-COD) methods predominantly rely on Convolutional Neural Networks (CNNs) or Vision Transformers (ViTs). Adhering to a discriminative pixel-wise classification paradigm, these models often struggle with limited generalization capabilities when confronted with challenging camouflaged datasets. In this paper, we present Generative Referring Camouflaged Object Detection (GRCOD) to address these limitations. Initially, GRCOD employs a robust pre-trained vision backbone to extract latent representations for both the camouflaged image and the corresponding ground truth mask. Subsequently, guided by a reference feature aggregation network (RFAN), a lightweight projection module learns the generative process from the image to the mask within the latent space. Finally, the model utilizes the pre-trained vision backbone to decode the predicted latent mask representation back into the image space, yielding the final segmentation mask for the camouflaged object. The design of GRCOD significantly reduces the number of trainable parameters, thereby mitigating the risk of overfitting. Furthermore, the effective utilization of pre-trained priors substantially enhances the model's generalization capability. Extensive experiments on standard benchmarks demonstrate that GRCOD significantly outperforms existing discriminative state-of-the-art methods.
</p>

## ⭐ Architecture
<p align="center">
    <img src="https://github.com/QuantumScriptHub/GRCOD/blob/main/data/GRCOD.png" alt="Architecture" />
</p>

<p align="justify">
The architecture design of our SimpleDiffusion. It consists of an Adaptive Cross-Modal Fusion Conditional Network for extracting multi-scale features as conditions, and a Latent Denoising Network for recovering clear mask predictions from the noised mask.
</p>

##  🚀 Modest Surprise
<p align="center">
    <img src="denoising-diffusion-pytorch/images/visulization.jpg" alt="Other Result" />
</p>

<p align="justify">
The comparison results in the figure above, from left to right, are Image, Depth, GT, Ours, CalibNet, M2For, and RDPNet. Clearly, our SimpleDiffusion not only achieves excellent detection results in the RGB-D salient object generation domain but also delivers outstanding segmentation performance in RGB-D instance segmentation.
</p>

## ⬇ Datasets
**All datasets are available in public**.
* Download the LFSD from [Here](https://www.eecis.udel.edu/~nianyi/LFSD.htm)
* Download the NJUD from [Here](https://pan.baidu.com/s/1o-kOaDVqjV_druBHjD3NAA)
* Download the NLPR from [Here](https://pan.baidu.com/s/1pocKI_KEvqWgsB16pzO6Yw)
* Download the DUTLF from [Here](https://pan.baidu.com/s/1mhHAXLgoqqLQIb6r-k-hbA)
* Download the SIP from [Here](https://pan.baidu.com/s/14VjtMBn0_bQDRB0gMPznoA)
* Download the VT5000 from [Here](https://pan.baidu.com/s/196S1GcnI56Vn6fLO3oXb5Q) with password:y9jj
* Download the VT821 from [Here](https://drive.google.com/file/d/0B4fH4G1f-jjNR3NtQUkwWjFFREk/view?resourcekey=0-Kgoo3x0YJW83oNSHm5-LEw)
* Download the VT1000 from [Here](https://drive.google.com/file/d/1NCPFNeiy1n6uY74L0FDInN27p6N_VCSd/view)
  
## 🛠️  Dependencies
```bash
* pip install -r requirements.txt
```
## 📦 Checkpoint cache

By default, our [checkpoints](https://drive.google.com/file/d/1OynVRx5rY8IM0UwlIxEKVrH_ujcsOIlY/view?usp=drive_link)  are stored in Google Drive. 
You can click the link to download them and proceed directly with inference. 

## ⚙ Configurations

#### Training

```shell
accelerate launch train.py --config config/model.yaml --num_epoch=150 --batch_size=32 --num_workers=4 --results_folder './results'
```

#### Inference 
```shell
accelerate launch sample.py \
  --config config/model.yaml \
  --results_folder ${RESULT_SAVE_PATH} \
  --checkpoint ${CHECKPOINT_PATH} \
  --num_sample_steps 10 \
  --target_dataset NJU2K \
  --time_ensemble
```

## 💻 Testing on your images
### 📷 Prepare images
If you have images at hand, skip this step. Otherwise, download a few images from [Here](https://pan.baidu.com/s/1o-kOaDVqjV_druBHjD3NAA).


## 🎫 License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)



