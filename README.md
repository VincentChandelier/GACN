
# Learned Focused Plenoptic Image Compression with Microimage Preprocessing and Global Attention
Pytorch implementation of the paper "Learned Focused Plenoptic Image Compression with Microimage 
Preprocessing and Global Attention". TMM2023.
This repository is based on [CompressAI](https://github.com/InterDigitalInc/CompressAI). 
We kept scripts for training and evaluation, and removed other components. 
 For the official code release, see the [CompressAI](https://github.com/InterDigitalInc/CompressAI).

## About
This repo defines the focused plenoptic image dataset “FPI22k" and the global-attention-based models for learned focused plenoptic image compression in "Learned Focused Plenoptic Image Compression with Microimage 
Preprocessing and Global Attention".

## dataset
“FPI2k” is  a focused plenoptic image dataset with 1910 images captured from real 
scenes indoor and outdoor with object depth variations.
From a single plenoptic image, 5×5 sub-aperture images can be 
generated with much larger disparities one from the other.
1910 focused plenoptic images are captured and manually annotated 
to 32 categories based on their contents

The original focused plenoptic images are available to download.()

![](./assets/FPI2k.png)
### Data preprocessing
Based on the observations that inter-microimage pixels, 
boundary incomplete microimages, and vignetting pixels in the 
microimages are ineffective in light field applications, like 
refocusing, multi-view rendering, etc., a sub-aperture images
lossless preprocessing scheme is proposed to reshape the 
sub-aperture effective pixels in each microimage and align the 
cropped microimages to the rectangular grid to be compatible 
with patch-based training and to reduce the pixel redundancy.
![](./assets/preprocessing.png)

The preprocessed focused plenoptic images are available to download.()

### Rendering
The dirctory Rendering provided the rendering code to render sub-aperture images from
 orignal or preprocessed focused plenoptic images

Run the *./Rendering/Original2SAI.m* to rendering the sub-aperture images from original focused plenoptic images.

Run the *./Rendering/Original2Preprocessed.m* to preprocess the original focused plenoptic images to preprocessd 
focused plenoptic images.

Run the *./Rendering/Preprocessed2SAI.m* to rendering the sub-aperture images from preprocessd focused plenoptic images.

## Global Attention Compression Network (GACN)
![](./assets/Network.png)
### Installation

Install [CompressAI](https://github.com/InterDigitalInc/CompressAI) and the packages required for development.
```bash
conda create -n FPIcompress python=3.9
conda activate FPIcompress
pip install compressai==1.1.5
pip install ptflops
pip install einops
pip install tensorboardX
```

> **Note**: wheels are available for Linux and MacOS.

## Usage

### traing dataset 
The 75080 patches of preprocessed plenoptic images ara available.
The full resolution test images are available.

### Training
An examplary training script with a rate-distortion loss is provided in
`train.py`. 

Training a model:
```bash
python train.py -d ./dataset --model Proposed -e 50  -lr 1e-4 -n 8  --lambda 1e-1 --batch-size 4  --test-batch-size 4 --aux-learning-rate 1e-4 --patch-size 384 384 --cuda --save --seed 1926 --clip_max_norm 1.0 --gpu-id 1 --savepath  ./checkpoint/PLConvTrans01
```

### Evaluation

To evaluate a trained model, the evaluation script is:

```bash
python Inference.py --dataset /path/to/image/folder/ --output_path /path/to/reconstruction/folder/ -m Proposed -p ./updatedCheckpoint/PLConvTrans01.pth.tar --patch 384
```


### Pretrained Models
Pretrained models (optimized for MSE) trained from focused plenoptic image patches are available.

| Method | Lambda | Link                                                                                              |
| ---- |--------|---------------------------------------------------------------------------------------------------|
| Proposed | 0.1 | []()    |
| Proposed | 0.05  | []()     |
| Proposed | 0.025 | []() |
| Proposed | 0.01 | []()    |
| Proposed | 0.005 | []() |
| Proposed | 0.013  | []()  |

All the checkpoints are available at

## Results

### Visualization

![visualization](/assets/Visualization.png)
>  Visualization of the reconstructed central sub-aperture image of "Car".

### RD curves
 Visualization of the reconstructed image Car.
 RD curves on I01 "Cars"
![original_rd](./assets/RdcurveOriginal.png)
![preprocessing_rd](./assets/RdcurvePreprocessed.png)
![reordering_rd](./assets/RdcurveReordering.png)

## Citation
```

```

## Related links
 * CompressAI: https://github.com/InterDigitalInc/CompressAI


