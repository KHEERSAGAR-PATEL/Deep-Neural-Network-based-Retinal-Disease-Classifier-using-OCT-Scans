# Deep-Neural-Network-based-Retinal-Disease-Classifier-using-OCT-Scans
Deep Neural Network-based Retinal Disease Classifier using OCT Scans
Implemented a CNN with six convolutional blocks in PyTorch to classify Diabetic Macular Edema (DME),
Choroidal NeoVascularization (CNV), and drusen from OCT images. Achieved over 99% training accuracy
and 96.30% testing accuracy on around 83,484 images. Utilized Grad-CAM for disease region visualization,
with the model using only 6.9% of the parameters of ResNet-50, making it suitable for real-time applications.

This document provides a detailed overview of the convolutional neural network (CNN) architecture used for medical image classification, particularly for classifying OCT images into categories like CNV, DME, Drusen, and Normal. Here's a structured summary:

### 3.1. Architecture
1. **Convolutional Neural Network (CNN)**:
   - Initially introduced by Alex Krizhevsky in 2012 during the ILSVRC competition, CNNs have significantly advanced pattern recognition in computer vision, making them ideal for medical imaging applications.
   - Key elements in a CNN include the convolution operation, defined by:
     \[
     y[n] = \sum_{k=-\infty}^{\infty} x[k]h[n - k]
     \]
   - 2D convolution (used for image data) involves parameters such as kernel size, stride, padding, and the number of input and output channels, which affect the model's capacity to learn local and global features.

2. **Regularization Techniques**:
   - **Batch Normalization**: Normalizes layer activations, potentially stabilizing the learning process by adjusting inputs to each layer.
   - **Weight Decay**: A form of regularization that penalizes large weights, reducing overfitting by modifying the cost function.
   - **Dropout**: Randomly deactivates neurons during training, reducing the risk of overfitting by forcing the network to rely on different subsets of neurons for different iterations.

3. **Backpropagation**:
   - The model's learning relies on adjusting weights to minimize the loss function using optimization algorithms such as SGD, RMSProp, and ADAM.
   
4. **Activation Functions**:
   - Activation functions like ReLU introduce non-linearity, essential for the CNN's ability to learn complex patterns.

5. **Downsampling Techniques**:
   - **Max Pooling** and **Average Pooling**: Both reduce spatial dimensions to decrease computational complexity and increase receptive fields, with max pooling selecting the highest value in a region, and average pooling computing the mean.

### 3.2. Implementation
- The architecture (OctNET) involves six convolutional blocks with ReLU activations and max pooling, followed by an average pooling layer. 
- This feature extractor generates a 512×1 feature vector, fed into a three-layer MLP with a dropout factor of 0.5.
- **Cross-Entropy Loss** is used for optimization, with weight adjustments made based on class imbalances.

### 3.3. Dataset
- The dataset consists of 83,484 training images and 968 testing images for OCT classification across four classes (CNV, DME, Drusen, Normal).
- Class imbalances are addressed by assigning weight factors to different classes during training, improving classification fairness across categories.



---

# Medical Image Segmentation with Modified U-Net Architecture

This project implements a modified U-Net architecture for semantic segmentation of medical images, designed to improve diagnostic accuracy and interpretability in medical imaging. Built with PyTorch, this model aims to segment images into meaningful regions, such as organs or tumor areas, with high precision and scalability across various medical imaging modalities (e.g., CT scans, MRI). 

## Overview

U-Net is a widely used deep learning architecture for image segmentation. It employs a symmetrical "U"-shaped structure with encoding and decoding paths. The encoding path captures features through convolutional and max-pooling layers, while the decoding path reconstructs feature maps to the original resolution using up-convolutions and skip connections. This enables the model to retain fine spatial details and effectively segment intricate regions within medical images.

This project extends the U-Net with several architectural improvements for handling medical images, leveraging a ResNet-34 encoder to further enhance feature extraction. Additionally, it includes a new scoring mechanism for better accuracy in pixel-wise segmentation tasks.

### Key Features
- **Modified U-Net Architecture**: Incorporates ResNet-34 encoder for more refined feature extraction.
- **Skip Connections**: Combines high-resolution details from the encoding path with the decoding path for precise segmentation.
- **Efficient Parameter Utilization**: Optimized design to maximize performance while maintaining computational efficiency.
- **Fine-Tuned for Small Object Segmentation**: Effective for tasks requiring the segmentation of small objects or regions in medical images.
- **High Transferability**: Adaptable to various medical imaging datasets, enhancing usability in diverse healthcare applications.

### Applications
- **Organ Segmentation**: Localizing organs or anatomical regions for preoperative planning.
- **Tumor and Lesion Detection**: Segmenting tumor regions for diagnostic and therapeutic procedures.
- **Artery and Retinal Layer Segmentation**: Identifying vascular structures and retinal layers in OCT and MRI.

## Architecture

The modified U-Net is a complete convolutional encoder-decoder network with symmetrical encoding and decoding paths connected by skip connections. Key architectural components include:

- **Encoder**: Composed of convolutional layers (3x3 filters) with max-pooling layers for downsampling, using padding to maintain spatial dimensions.
- **Decoder**: Mirror structure of the encoder with up-sampling layers and transposed convolutions to restore resolution and retain spatial details.
- **Skip Connections**: High-resolution features from the encoder are concatenated with up-sampled feature maps, enhancing context preservation in the decoding path.
- **Scoring Mechanism**: Additional 1x1 convolutions at each decoding block to refine the segmentation map with a higher accuracy score.

The architecture has been shown to improve performance in comparison to standard U-Net models, with better retention of spatial information and robustness in segmenting small objects.

## Dataset

The model was trained on a medical imaging dataset containing 600 OCT images (from CT or MRI scans). The images are segmented into classes, typically representing different tissue types (e.g., tumor vs. normal). The dataset has been preprocessed for training and evaluation, including resizing and normalization.

### Evaluation Metrics

- **Accuracy**: Measures the overall correctness of segmentation.
- **Mean Intersection over Union (mIoU)**: Evaluates the overlap between predicted and actual segmentations, providing a detailed assessment of segmentation accuracy.


### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- OpenCV, NumPy, and other standard libraries as listed in `requirements.txt`


## Results

The modified U-Net architecture demonstrated enhanced accuracy and mIoU scores, showing promise for practical applications in medical diagnostics. Comparative evaluation against DeepLabV3Plus and U-Net++ benchmarks indicates competitive performance and robustness to noise in imaging data.
