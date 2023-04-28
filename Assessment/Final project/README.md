# Revolutionizing Beverage Packaging Waste: AI-Powered Classification for Efficient Sorting and Recycling

### Research Question

Can an AI-powered beverage packaging classification system improve the accuracy and efficiency of sorting recyclable materials in the beverage industry compared to traditional sorting methods?



### Project Overview


My project experimented with image classification and object detection models. The image classification application takes an image as an input and outputs the type of drinking waste from the image: 


<img width="200" alt="image" src="https://github.com/LeoLiu5/casa0018-final-project/blob/main/Assessment/Final%20project/Edge%20Impulse/Transfer%20learning%20for%20image%20classification/Trial%201/5.png">



The object detection application takes an image as an input and outputs the types, numbers, positions, and sizes of drinking wastes from the image:



<img width="200" alt="image" src="https://github.com/LeoLiu5/casa0018-final-project/blob/main/Assessment/Final%20project/Edge%20Impulse/Object%20Detection/FOMO-testing%20learning%20rate/Figure%204.png">


Both applications share a similar cyclical workflow and use cameras from mobile devices to capture images:

<img width="400" alt="image" src="https://github.com/LeoLiu5/casa0018-final-project/blob/main/Assessment/Final%20project/else/Figure%203.png">


### Application Design


I started the project by exploring pre-trained models from Edge Impulse, which guides users through collecting data, training machine learning models, and deploying them on edge devices.

The transfer learning for image classification applications allows MobileNet to learn features from the large-scale dataset with common objects hence making it faster and more accurate to tune or adapt to new tasks. MobileNet is a complex and powerful CNN architecture designed for mobile devices.

<img width="700" alt="image" src="https://github.com/LeoLiu5/casa0018-final-project/blob/main/Assessment/Final%20project/else/Figure%2010.png">



To understand the building blocks of the models and evaluate the effects of more hyper-parameters, I browsed the Keras (expert) mode and structured a basic Keras model in Colab. The model is a representative convolutional neural network (CNN) architecture commonly used for image classification.



For object detection, I trained Edge Impulse FOMO (Faster Objects, More Objects), which has the same architecture as an image classification model without the final convolution layers. 




### Data available

For this project, I downloaded the Drinking Waste Classification dataset, which was collected with a 12 MP phone camera, from Kaggle (https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification). 


<img width="500" alt="image" src="https://github.com/LeoLiu5/casa0018-final-project/blob/main/Assessment/Final%20project/else/Figure%206.png">


The raw images are then uploaded to Edge Impulse and preprocessed using the image processing block. Edge Impulse allows the images to be labelled manually and split into training and test data. The image processing block provides various image preprocessing options, such as resizing, cropping, changing colour depth, and normalization. 



In addition to the drinking waste images, I uploaded both black-and-white and RGB background images to Edge Impulse to train the model to identify any background or noise from the input images: 


<img width="400" alt="image" src="https://github.com/LeoLiu5/casa0018-final-project/blob/main/Assessment/Final%20project/else/Figure%207.png">


Grayscale Textures Classification dataset (https://github.com/abin24/Textures-Dataset).

<img width="600" alt="image" src="https://github.com/LeoLiu5/casa0018-final-project/blob/main/Assessment/Final%20project/else/Figure%208.jpeg">

RGB Describable Textures dataset (https://www.robots.ox.ac.uk/~vgg/data/dtd/).


### Outcomes Anticipated

I anticipated MobileNet V2 to perform better than V1 because V2 has more layers and an inverted residual structure while using more RAM and ROM for better results. I also anticipated the basic Keras model to provide a lower accuracy than MobileNet due to the absence of Batch Normalization and fine-tuning layers. Finally, I anticipated adding a grayscale or RGB "Other/Background" classification will improve model accuracy.



## Documentation of experiments and results 


 
Model training results, description of training runs, and model architecture choices for image classification using Edge Impulse can be found in the ["Transfer learning for image classification" folder.](https://github.com/LeoLiu5/casa0018-final-project/tree/main/Assessment/Final%20project/Edge%20Impulse/Transfer%20learning%20for%20image%20classification)
 
Model training results, description of training runs, and model architecture choices for the basic Keras model using Colab can be found in the ["The Keras Sequential model" folder.](hhttps://github.com/LeoLiu5/casa0018-final-project/tree/main/Assessment/Final%20project/The%20Keras%20Sequential%20model)
 
Model training results, description of training runs, and model architecture choices for object detection using Edge Impulse can be found in the ["Object Detection" folder.](https://github.com/LeoLiu5/casa0018-final-project/tree/main/Assessment/Final%20project/Edge%20Impulse/Object%20Detection)

The visual record of experiments can be found in the ["Real-world deployment" folder.](https://github.com/LeoLiu5/casa0018-final-project/tree/main/Assessment/Final%20project/Real-world%20deployment)

## Critical reflection and learning from experiments 

### Observations from Experiments

MobileNet V2 performed better than V1, proving that V2 is a refinement of V1 that makes it more efficient and powerful. MobileNetV2 160x160 0.5 has better performance than 96x96 0.35. It could be the increase in image size or change in colour depth, or 0.5 uses more RAM and ROM in optimizations. By subtracting the background or noise, the model can focus on object features and improve the accuracy of the classification process.

From the Keras sequential model in Colab, optimizer Adam (Adaptive Moment estimation) presented the best training results. Nadam is less accurate but requires less memory. 320x320 px images provided the best accurate training results.

Almost all the validation accuracies are lower than the training accuracies because only the training accuracies of the unoptimized (float32) models and the validation accuracies of the quantized (int8) models are recorded. Int8 has lower performance because quantization reduces the precision of model weights and activations, resulting in a loss of information and accuracy.



### Factors Influencing Results

The number of epochs, the number of neurons, batch size, and the learning rate require careful consideration. It is inaccurate to assume that the higher their values are, the better the results. Once their most suitable values are surpassed, the model performance will decrease or present no significant changes.

Other important factors that need to be considered are image size and resolution, optimizer, and image background.

### Weaknesses 

Most challenges during this project were caused by the large dataset, especially in RGB or 320x320 px, which:
- exceeds training time limits; 
- forces to reduce the number of epochs; 
- becomes unavailable to many models;
- requires too much time on labelling items in Object Detection mode.

The large-scale dataset and the limitations of the hardware or technologies caused challenges. Some models were ignored because they exceeded training time limits or memory capacity. 

In addition, the variation of beverage packaging is quite large in terms of colours, shapes, or sizes. Sometimes, even humans have trouble distinguishing between plastic bottles and glass bottles.

 


### Potential Improvements

In the future, it is worth experimenting with more potential models (VGGNet, ResNet, InceptionNet), layers, or convolution blocks with Tensorflow. EON Tuner in Edge Impulse can be used to find the most optimal architecture. The training dataset needs to be decided carefully. In the case of identifying distinctive shapes or textures of beverage packaging, it is important to carefully choose a diverse set of images that represent the different types of packaging.

I will use CCD cameras and advanced algorithms following current research to develop a model since now I have a basic understanding of deep learning.

### Feedback from Reviews 

"A great project with lots of experimentation and analysis. I liked your introduction to the problem and you showed you had done some research into the problem. You highlight that you used a Kaggle dataset and added in some of your background images, in your report I would recommend you show some examples of both of these to help the reader understand the source data."

"You have a clear research question "Classification of objects for efficient recycling using computer vision" and back up your project by showing some literature on the subject. You decide to use a Kaggle dataset and do multiple trials to test how different parameters such as image size, learning rate, and number of neurons, influence your model accuracy. You are discussing different trials and show multiple graphs to explain your trials."

"Using texture images as background dataset is a good approach. Constructed a basic Keras model and experimented with the best optimizers. Tried with object detection. The reflection is effective. Did not compare the training loss and validation loss which might solve overfitting."

"Model performs a high accuracy; Included many model's trials using different parameters; Exploration of Neural networks in Keras mode. This is a presented and thoughtful project."
