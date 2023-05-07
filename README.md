# ImageColorization
Image Colorization with GAN, Ara Guler. 

### Here is our code's explanation:

1.Imports the necessary libraries for loading and transforming images, setting up a data pipeline using PyTorch, and visualizing images using matplotlib.

2.Mounts the user's Google Drive to Colab.

3.Loads a subset of images from a given directory and randomly splits them into training and validation sets.

4.Defines a MakeDataset class which inherits from PyTorch's Dataset class. The __init__ method of this class applies some random transformations to the images (resizing, horizontal flipping). The __getitem__ method loads an image, converts it to the RGB color space, applies the defined transformations, converts the image to the LAB color space, and rescuescales the L and ab channels. The __len__ method returns the number of images in the dataset.

5.Instantiates the MakeDataset class for the training and validation sets and creates PyTorch DataLoader objects for them.

6.Defines a function lab_to_rgb which converts L and ab tensors back to the RGB color space.

7.Loads a batch of data from the training set, unpack the L and ab channels, and visualizes the L channel and the corresponding ground truth and reconstructed RGB images using matplotlib.

### Packages we have used in the code:

- torch: PyTorch is a deep learning framework for training and deploying neural networks.
- torchvision: A package that contains utilities for loading and transforming images, and for using pre-trained models.
- numpy: A package for scientific computing with Python. It provides support for large, multi-dimensional arrays and matrices, and a large library of mathematical functions to operate on these arrays.
- glob: A module that provides support for filename globbing (wildcard matching).
- matplotlib: A plotting library for creating static, animated, and interactive visualizations in Python.
- PIL: The Python Imaging Library adds image processing capabilities to your Python interpreter.
- skimage: A package for image processing in Python. It provides algorithms for image processing and computer vision tasks, including denoising, color space conversion, and object detection.
- tqdm: A package that provides a progress bar for loops and other iterators.
- datetime: A module that provides support for working with dates and times.
- fastai: A library that provides a high-level API for training and deploying deep learning models. It is built on top of PyTorch.

### Technologies we have used:

1.Google Collab: for training and predicting the data.

2.Jupter Notebook: Python development environment.

3.Github: To store our data in a repository.

### Training function:

Our our main train function contains input arguements as
follows:

- discModel: the discriminator model
- genModel: the generator model
- loader: a data loader that provides a stream of training data
- optimizerForDiscriminator: an optimizer for the discriminator
model
- optimizerForGenerator: an optimizer for the generator model
- L1Loss: a loss function for computing the L1 loss
- BCELoss: a loss function for computing the binary cross
entropy loss
- generatorScaler: an object for scaling the loss of the generator
during training
- discriminatorScaler: an object for scaling the loss of the
discriminator during training
<br>

### Examples from our work:

-Data scraping:
![image](https://user-images.githubusercontent.com/73427323/236674856-c568ea4b-e488-48d5-ae1d-309050c1d1fa.png)

-Prediction:
![prediction](https://user-images.githubusercontent.com/73427323/236674870-fb39f40e-effd-466e-a6fd-8c63aabcbaa0.png)

### Comparison:
We want to compare our balanced dataset with pretrained datasets. At the top left you can see our output and at the bottom you can see the pretrained models prediction accuracy
![example1](https://user-images.githubusercontent.com/73427323/236674961-e45f72b5-f96b-4d84-a00a-0aee0c37fc15.png)
![example2](https://user-images.githubusercontent.com/73427323/236674963-26e7e5b0-8c09-4230-aa19-7aa6748b1e1b.png)


## This project is made by Ahmet Kaan Memioğlu, Şükrü Erim Sinal and Emrecan Üzüm
