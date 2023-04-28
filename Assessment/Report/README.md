# Revolutionizing Beverage Packaging Waste: AI-Powered Classification for Efficient Sorting and Recycling

Leo Liu

My GitHub repository with all the project work: https://github.com/LeoLiu5/casa0018-final-project.

Edge Impulse project links: 

Beverage Packaging Waste Classification - Transfer learning for image classification: https://studio.edgeimpulse.com/public/184551/latest.

Beverage Packaging Waste Classification - Object Detection: https://studio.edgeimpulse.com/public/193760/latest.

## Introduction

In recent years, beverage packaging waste has become a significant environmental issue. Fortunately, AI-powered classification for efficient sorting and recycling of beverage packaging waste address this issue (Fig 1). By training deep learning models to detect different types of materials based on their shape, degradation level, size, colour, and contamination level, objects can be accurately identified or sorted (Ahmed & Asadullah, 2020).

Near-infrared reflectance measurement is an effective method to identify plastic bottle composition (Tachwali et al., 2007). The NIR reflectance spectra of different plastic types exhibit distinct differences, allowing for the accurate classification of bottles based on their chemical composition. In addition to composition classification, bottle colour can be detected from a charged-coupled device (CCD) camera, and a fusion of quadratic discriminant analysis (QDA) and tree classifiers (Fig 2).

Classifying packaging wastes is a challenging assignment which requires advanced technologies and equipment. Because of my limited experience with machine learning and hardware optimization, I used Edge Impulse to learn how to train image classification and object detection models to accurately and efficiently sort beverage packaging wastes, including aluminum cans, glass bottles, plastic bottles, and milk containers. The real-world applications were deployed on mobile devices.

<img width="500" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 1.png">

Figure 1. An AI-powered robotic system (Ahmed & Asadullah, 2020).

<img width="500" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 2.png">

Figure 2. QDA and tree classifiers ensure the accuracy of colour detection, while the CCD camera captures high-quality images for analysis (Tachwali et al., 2007).

## Research Question

Can an AI-powered beverage packaging classification system improve the accuracy and efficiency of sorting recyclable materials in the beverage industry compared to traditional sorting methods?

## Application Overview

The image classification application takes an image and outputs the drinking waste type (Fig 3). The object detection application takes an image and outputs drinking waste type, number, position, and size (Fig 4). Both applications share a similar cyclical workflow and use cameras from mobile devices to capture images (Fig 5).

The input images are preprocessed, which involves transforming raw images into a format that can be effectively processed by machine learning models, to ensure that the images are of a consistent size and resolution. This block uses Digital Signal Processors (DSPs) specializing microprocessors designed to perform mathematical operations on digital signals (Edge Impulse Documentation, 2022a). DSPs perform various signal processing operations, such as filtering, signal generation, modulation or demodulation, and compression or decompression. 

Feature extraction involves identifying and extracting relevant features such as shape, colour, and texture from the preprocessed images. These features train models to recognize and classify different objects or patterns in the images. The output is the classified beverage packaging type, which is used to sort the recyclable materials.

<img width="200" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/Edge Impulse/Transfer learning for image classification/Trial 1/5.png">

Figure 3. Image classification mobile application.

<img width="200" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/Edge Impulse/Object Detection/FOMO-testing learning rate/Figure 4.png">

Figure 4. Object detection mobile application.

<img width="400" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 3.png">

Figure 5. Application workflow.

## Data

I downloaded the Drinking Waste Classification dataset collected with a 12 MP phone camera from Kaggle (Fig 6). The raw images are then uploaded to Edge Impulse and preprocessed using the image processing block, which allows the images to be labelled manually and split into training and test data. The image processing block provides various image preprocessing options, such as resizing, cropping, changing colour depth, and normalization (Edge Impulse Documentation, 2022a).

I included background images to identify background or noise from the input images. I suspected the RGB backgrounds from the training images were too colourful and distractive. Therefore, I downloaded a black-and-white image dataset of surface texture and created a grayscale "Other/Background" classification (Fig 7). I also downloaded another colourful image dataset of describable textures and experimented with an RGB "Other/Background" classification (Fig 8). I anticipated adding a grayscale or RGB "Other/Background" classification will improve model accuracy.

<img width="500" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 6.png">

Figure 6. Drinking Waste Classification dataset (https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification).

<img width="400" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 7.png">

Figure 7. Grayscale Textures Classification dataset (https://github.com/abin24/Textures-Dataset).

<img width="600" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 8.jpeg">

Figure 8. RGB Describable Textures dataset (https://www.robots.ox.ac.uk/~vgg/data/dtd/).



## Model

I started the project by exploring pre-trained models from Edge Impulse, which guides users through collecting data, training machine learning models, and deploying them on edge devices. 

The transfer learning for image classification applications allows MobileNet to learn features from the large-scale dataset with common objects hence making it faster and more accurate to tune or adapt to new tasks (Edge Impulse Documentation, 2022b). MobileNet V2 has more layers and an inverted residual structure while using more RAM and ROM for better results (Fig 9). 

To understand the building blocks of the models and evaluate the effects of more hyper-parameters, I browsed the Keras (expert) mode and structured a basic Keras model in Colab. The model has a representative convolutional neural network (CNN) architecture commonly used for image classification (Fig 10). It has a lower accuracy than MobileNet due to the absence of Batch Normalization and fine-tuning layers (Tensorflow Core, 2022).

Finally, I tried Edge Impulse FOMO (Faster Objects, More Objects), which has the same architecture as an image classification model without the final convolution layers (Fig 11). 



<img width="700" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 10.png">

Figure 9. MobileNet is a complex and powerful CNN architecture designed for mobile devices (Hollemans, 2018).


<img width="400" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 11.png">

Figure 10. Three convolutional blocks followed by max-pooling layers are designed to extract important features from the input image while reducing the spatial dimensions of the feature maps to make computation efficient.

<img width="400" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 12.png">

Figure 11. Unlike other object detection algorithms outputting bounding boxes, FOMO outputs a heatmap and then highlights the objects, consuming up to 30x less processing power and memory than MobileNet SSD or YOLOv5 (Edge Impulse Documentation, 2022c).


## Experiments

My experiment started with image classification, and I measured each model's performance based on their overall training, validation, and test accuracies (Fig 12) (Fig 13). I also gathered some testing objects to measure the real-world deployment performance for each model (Fig 14). Data augmentation and auto-balance datasets are enabled because of the large and unbalanced dataset.

In trials 0 and 1, I compared MobileNetV1 and V2. In trial 2, I tested the colour depth effect by changing the grayscale to RGB. In trial 3, I added grayscale background images and created an "Other/Background" classification to examine the effects of identifying backgrounds from input images (Fig 7). In trials 4, 5, and 6, I evaluated multiple hyper-parameters (Fig 15). A training cycle or an epoch represents one iteration over the entire dataset. Increasing the number of neurons will add more complexity to the neural network architecture, while a simple neural network architecture will train faster (Edge Impulse Documentation, 2022e). The learning rate controls the updating frequency of the internal parameters or how fast the neural network learns (Edge Impulse Documentation, 2022b). In trial 7, I upgraded the pre-trained model and anticipated better results. In trial 8, I replaced the grayscale background images with RGB background images (Fig 8).

To evaluate the effects of more hyper-parameters, I created a Keras sequential model in Colab and recorded each trial on Weights & Biases (WandB) (Fig 16). WandB is a Python package allowing model training sessions from Colab to be monitored in real-time and generating an experiment report (https://wandb.ai/leoliu11/projects) (Liu, 2023). Keras Optimizer trains the model to find an optimized loss function and desired weights (EDUCBA, 2023). The dataset is divided into batches because the model cannot process all images at once.

Finally, I experimented with Edge Impulse FOMO, and measured model performance based on F1 score and test accuracy (Fig 17) (Fig 18) (Fig 19) (Fig 20).

<img width="450" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 13.png">

Figure 12. Image classification data split visualization.

<img width="900" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 14.png">

Figure 13. Results table.

<img width="500" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Test objects for real-world deployment 1.jpeg">

Figure 14. Testing objects.

<img width="650" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 16.png">

Figure 15. Hyper-parameters.


<img width="450" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 17.png">

Figure 16. The Keras Sequential model data split visualization.

<img width="450" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 18.png">

Figure 17. Object detection data split visualization.

<img width="300" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 17.5.png">

Figure 18. Labelling each image using bounding boxes from the labelling queue.

<img width="900" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure18.png">

Figure 19. Results table.

<img width="600" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure19.png">

Figure 20. Hyper-parameters.

## Results and Observations


MobileNetV2 performed better than V1, proving that V2 is a refinement of V1 that makes it more efficient and powerful (Fig 13). MobileNetV2 160x160 0.5 has better performance than 96x96 0.35. It could be the increase in image size or change in colour depth, or 0.5 uses more RAM and  ROM in optimizations (Fig 13). 
The number of epochs, the number of neurons, batch size, and the learning rate require careful consideration. It is inaccurate to assume that the higher their values are, the better the results. Once their most suitable values are surpassed, the model performance will decrease or present no significant changes (Fig 15) (Fig 20) (Fig 21). 

By subtracting the background or noise, the model can focus on object features and improve the accuracy of the classification process (Fig 22).

From the Keras sequential model in Colab, optimizer Adam (Adaptive Moment estimation) presented the best training results (Fig 23). Nadam is less accurate but requires less memory (Fig 24). 320x320 px images provided the best accurate training results (Fig 25) (Fig 26).


Almost all the validation accuracies are lower than the training accuracies because only the training accuracies of the unoptimized (float32) models and the validation accuracies of the quantized (int8) models are recorded. Int8 has lower performance because quantization reduces the precision of model weights and activations, resulting in a loss of information and accuracy (Edge Impulse Documentation, 2022d).

The large-scale dataset and the limitations of the hardware or technologies caused challenges. Some models were ignored because they exceeded training time limits or memory capacity. In the future, I will use EON Tuner to experiment with more models (VGGNet, ResNet, InceptionNet), convolution blocks or layers. The dataset needs to be updated with more diverse and representative images because the variation of beverage packaging is large in terms of colours, shapes, or sizes. I will use CCD cameras and advanced algorithms following current research to develop a model since now I have a basic understanding of deep learning (Fig 2).
















<img width="600" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/WechatIMG72.png">

Figure 21. The default batch size 32 provided the most accurate results.

<img width="800" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 19.png">

Figure 22. The model recognizes backgrounds instead of identifying everything as bottles.

<img width="600" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 22.png">

Figure 23. Adam is most popular in neural networks and useful when the dataset and the number of parameters are large (EDUCBA, 2023).

<img width="600" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 23.png">

Figure 24. N stands for Nesterov and it is more efficient than the previous implementations (EDUCBA, 2023).


<img width="600" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 25.png">

Figure 25. Increases in image resolution increase model performance (Thambawita et al., 2021).

<img width="600" alt="image" src="/Users/leoliu/Documents/GitHub/casa0018-final-project/Assessment/Final project/else/Figure 26.png">

Figure 26. High-resolution images require more processing memory and power.

## Bibliography

1. Ahmed, A.A. and Asadullah, A.B.M. (2020) “Artificial Intelligence and machine learning in Waste Management and recycling,” Engineering International, 8(1), pp. 43–52. Available at: https://doi.org/10.18034/ei.v8i1.498. 

2. Edge Impulse Documentation (2022a) Impulse design, Edge Impulse. Available at: https://docs.edgeimpulse.com/docs/edge-impulse-studio/impulse-design (Accessed: March 26, 2023). 

3. Edge Impulse Documentation (2022b) Transfer learning (Images), Edge Impulse. Available at: https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/transfer-learning-images (Accessed: March 26, 2023). 
 
4. Edge Impulse Documentation (2022c) FOMO: Object detection for constrained devices, Edge Impulse. Available at: https://docs.edgeimpulse.com/docs/edge-impulse-studio/learning-blocks/object-detection/fomo-object-detection-for-constrained-devices (Accessed: March 26, 2023). 

5. Edge Impulse Documentation (2022d) Increasing model performance, Edge Impulse. Available at: https://docs.edgeimpulse.com/docs/tips-and-tricks/increasing-model-performance#large-difference-between-quantized-int8-and-unoptimized-float32-model-performances(Accessed: March 26, 2023). 
 
6. Edge Impulse Documentation (2022e) Lower compute time, Edge Impulse. Available at: https://docs.edgeimpulse.com/docs/tips-and-tricks/lower-compute-time#reduce-the-complexity-of-your-neural-network-architecture (Accessed: March 26, 2023). 
 
7. EDUCBA (2023) Keras optimizers: Types and models of Keras optimizers with examples, EDUCBA. Available at: https://www.educba.com/keras-optimizers/ (Accessed: March 18, 2023). 
 
8. Liu, L. (2023) Examine the Keras sequential model in Beverage Packaging Waste Classification, W&B. Available at: https://wandb.ai/leoliu11/Optimizers/reports/Examine-the-Keras-Sequential-model-in-Beverage-Packaging-Waste-Classification--VmlldzozNzkyNTU4 (Accessed: March 28, 2023). 

9. Tachwali, Y., Al-Assaf, Y. and Al-Ali, A.R. (2007) “Automatic multistage classification system for Plastic Bottles Recycling,” Resources, Conservation and Recycling, 52(2), pp. 266–285. Available at: https://doi.org/10.1016/j.resconrec.2007.03.008. 

10. Thambawita, V. et al. (2021) “Impact of image resolution on deep learning performance in Endoscopy Image Classification: An experimental study using a large dataset of endoscopic images,” Diagnostics, 11(12), p. 2183. Available at: https://doi.org/10.3390/diagnostics11122183. 

11. Tensorflow Core (2022) Transfer learning and fine-tuning, TensorFlow. Available at: https://www.tensorflow.org/tutorials/images/transfer_learning (Accessed: March 28, 2023).


----

## Declaration of Authorship

I, Leo Liu, confirm that the work presented in this assessment is my own. Where information has been derived from other sources, I confirm that this has been indicated in the work.


*Leo Liu*

21st April 2023.
