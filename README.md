# Multimodal MRI-based Synthetic Image Generation

## Abstract
> Pediatric high-grade gliomas remain a major challenge in the medical field as brain tumors are a leading cause of cancer-related death in children, with only 20% surviving past 5 years. Lack of sufficient MRI datasets of pediatric gliomas due to the high privacy restrictions and disease rarity impose major obstacles in training an accurate AI model for early detection and diagnosis. To combat this issue, multi-modal data augmentation using Denoising Diffusion Probabilistic and Convolutional GAN models is proposed to generate high quality synthetic MRI data. The Diffusion model was adapted to utilize a modified UNet architecture with increased layers and spatial complexity to accommodate the intricacies of the grayscale MRI images. The two models were then compared using Frechet Inception Distance score, with the redesigned Diffusion model obtaining a value of 35, and performing 70% better than the baseline GAN model, which received a score of 119. t-Distributed Stochastic Neighbor Embedding and Kernel Density Estimate distribution plots were graphed to visualize the similarities between real and generated images. After testing with T1w images, the Diffusion model was further modified to support multiple modalities of MRI including T2 and FLAIR. This allowed for the combination of these modalities into a three channel input and output Diffusion model capable of stacking them onto a single RGB image, providing deeper insights into tumor segmentation and variations between the different modalities.  Additionally, the modified Diffusion model was utilized to train several modern CNN models including ResNet50 and VGGNet to analyze the image classification accuracy.

## Goal & Further Motivation
Generating Synthetic data is extremely beneficial as it enables research into uncommon disease presentations and eliminates privacy concern providing increased diversity in dataset size of high-grade images. Expediting the time taken for the Brain MRI and review process is essential, as the delay time can cost a child their life. This project aims to provide a reliable and cost-effective solution to generate high-quality data for increasing diversity of datasets in order to solve the aforementioned problems. Below are the three main goals of the project.

1) Design and Train Multimodal (T1w, T2w, T2-FLAIR) Diffusion Networks on augmented pediatric clinical high-grade glioma (cancer) data to generate synthetic data at high volumes to improve Radiologist’s efficiency
2) Showcase comparable accuracy to real MRI scans using similarity scoring
3) Allow for greater diversity in the dataset to boost classification accuracy, through various augmentation and segmentation techniques, as well as addressing any privacy concerns. 

## Methodology

### Data Preprocessing
The input data set was preprocessed in a methodical fashion, similar to many common approaches:
1) Resize Image to resolution 128x128 and Shuffle Dataset
2) Convert images into Tensor Slices to rescale pixel values from [0,255] to [0,1]
3) Normalize Pixel Distribution using a Mean of 0.5 and Standard Deviation of 0.5

### Augmentations & Transformations
Affine transformations and histogram equalization methods were utilized to enhance dataset size and diversity. In addition, images were cropped and scaled through dynamic processing via data manipulation programming to improve model performance and further adjust the intensity to produce an augmented dataset of around 24x size.

## Denoising Diffusion Probabilistic Model (DDPM)

##### Model Selection (Note): As part of testing, a Deep Convolutional Generative Adversarial Network (DCGAN) was used as a control model to monitor the growth of the DDPM.

The Diffusion model takes in the noisy image $𝒙_𝒕$ from the Forward Diffusion Process and predicts the amount of noise that was added from the original image $𝒙_𝟎$ to result in $𝒙_𝒕.$

### Forward Diffusion
Forward diffusion is utilized on every image in each batch and each epoch. By adding noise, forward diffusion allows for us to create training data, which lets the model learn how to remove it.

### Reverse Diffusion
At each step of the process, a probability distribution is determined, consisting of the possible values of 𝒙_(𝒕−𝟏), and the variance is initialized to add randomness to reduce the likelihood of mode collapse.

### Adaptation to Multimodality
Using the Singular Modality Diffusion model, a new redesigned Diffusion model is created to accommodate the input of three channel inputs, with each channel corresponding to a different modality
