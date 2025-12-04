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
We use the pre-trained <a href="https://github.com/Stability-AI/stablediffusion"> Stable Diffusion Variational Auto-Encoder</a> to abtain the latent representation of input images and reconstruct the predicted segmentation mask from the latent space. 
For the reconstruction capability of SD VAE, please check the <a href="https://github.com/QuantumScriptHub/GRCOD/blob/main/data/ICME2025_Appendix.pdf">supplementary material</a>. The latent projection model was purely developed on CNN, and does not contain down-sampling layers to prevent information loss.
The figure above presents the detailed architecture of our GRCOD model, which comprises key components such as an Image Encoder, Image Decoder, Reference Feature Aggregation Network, and Latent Projection Module. These modules are designed to extract and aggregate features from the reference images, and then fuse and interact them with the features of the camouflaged target. In this way, the model is guided to generate target masks in the camouflaged image that are consistent with the characteristics of the reference features.
</p>

##  🚀 Modest Surprise
<p align="justify">
In extensive experiments, we further observe that our model exhibits strong generalization capability. Although we do not adopt the traditional denoising process of diffusion models, we can still effectively exploit the visual priors of the pretrained VAE in Stable Diffusion. As a result, the generalization ability of our model not only matches that of models obtained by directly fine-tuning Stable Diffusion, but its inference efficiency is also improved by approximately an order of magnitude. Therefore, we argue that, among generative approaches, GRCOD is not only state-of-the-art at present, but is also likely to remain competitive for a considerable period of time, as it is carefully designed to balance both segmentation quality and inference efficiency for Ref-COD.
</p>

## ⬇ Datasets
**Download  [R2C7K dataset](https://pan.baidu.com/share/init?surl=LHdqpD3w24fcLb_dbR6DyA) with access code 2013 on Baidu Netdisk.**
The following figure shows examples from the R2C7K dataset. Note that, in the Camo subset, the camouflaged objects are overlaid with their annotations in orange.
<p align="center">
    <img src="https://github.com/QuantumScriptHub/GRCOD/blob/main/data/r2c7k.png" alt="Dataset" />
</p>
  
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
python train.py --config ./configs/cod_train.yaml
```

#### Inference 
```shell
python valid.py --config ./configs/cod_valid.yaml
```

## 💻 Testing on your images
### 📷 Prepare images
If you have images at hand, skip this step. Otherwise, download a few images from [Here](https://pan.baidu.com/share/init?surl=LHdqpD3w24fcLb_dbR6DyA).


## 🎫 License

This work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

By downloading and using the code and model you agree to the terms in the  [LICENSE](LICENSE.txt).

[![License](https://img.shields.io/badge/License-Apache--2.0-929292)](https://www.apache.org/licenses/LICENSE-2.0)



