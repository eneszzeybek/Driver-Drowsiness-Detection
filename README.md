# Source of the Dataset

https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset

# Dataset Information

Our dataset includes 1452 images. 726 images for closed eyes, 726 images for opened eyes.

# Conclusion

We used the Pytorch framework to apply CNN in the dataset. 

In this project, we have two steps. Firstly, we train a model that uses binary classification CNN to label the eyes of drivers as opened or closed. Finally, applying a computer vision technique called dlib landmarks detection, we will analyze the eyes of the driver for ten seconds to check whether the driver is sleepy or not by using our CNN model in real-time. Taking the mean of the drowsiness during the ten seconds will tell us the drowsiness of the driver.
