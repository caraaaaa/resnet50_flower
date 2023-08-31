# 102 Flower Classification

## Overview
This project aims to fine-tune a ResNet50 model to classify flowers that are commonly occurring in the United Kingdom. A [previous study](https://www.robots.ox.ac.uk/~vgg/publications/2008/Nilsback08/)  utilized an SVM and a combination of features to achieve 72.8% accuracy. This project shows that a **93.04%** accuracy can be achieved with a more sophisticated deep learning model.

## Data Understanding
The [dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) contains 8189 images and 102 classes. Each class consists of between 40 and 258 images.

![distribution](./src/class_distribution.png)

As described in the [dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html)  
>Each images have large scale, pose and light variations. In addition, there are categories that have large variations within the category and several very similar categories.
- Flowers might change color throughout their lifespan.
- Flowers can deform in various ways.
- Many types of flowers share similar shapes and colors.

![lifespan](./src/class_image1.png)

![color](./src/class_image.png)

## Methods
A ResNet50 pretrained on [IMAGENET1K_V2](https://pytorch.org/vision/stable/models.html) is used. To fit the data into the model, images are resized and normalized. The FC layer is replaced so that it can produce scores for each of the 102 classes. Layers other than the FC layer are frozen.


## Performance
Predictions were evaluated using [balanced accuracy](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html). The model achieved a **93.04%** accuracy after 10 epochs of training. The model was trained on Google Colab using a T4 GPU.

![cmat](./src/cmat.png)

## Next Steps
- Dealing with overfitting problem (i.e. data augmentaion, L2 regularization, learning rate scheduler)
![curve](./src/learning_curve.png)