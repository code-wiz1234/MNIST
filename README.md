# MNIST
The MNIST dataset, comprises grayscale images of size 28x28 pixels. It serves as an ideal starting point for understanding the fundamentals of image classification and testing CNN architectures. The dataset is split into a 6:1 ratio for training and testing, allowing for the evaluation of model performance on unseen data.

1. Input Layer
Input Shape: (28, 28, 1) for grayscale MNIST images.
2. Convolutional Layers
Convolutional Layer 1:
Number of Filters: 32
Filter Size: 3x3
Stride: 1
Padding: 1 (Same Padding)
Activation Function: ReLU
Batch Normalization: Applied after this layer
Convolutional Layer 2:
Number of Filters: 32
Filter Size: 3x3
Stride: 1
Padding: 1 (Same Padding)
Activation Function: ReLU
3. Pooling Layer
Type: Max Pooling
Pool Size: 2x2
Stride: 2
4. Dropout Layer
Dropout Rate: 0.25 (25% of neurons are dropped)
Batch Normalization: Applied after this layer
5. Additional Convolutional Layers
Convolutional Layer 3:
Number of Filters: 64
Filter Size: 3x3
Stride: 1
Padding: 1 (Same Padding)
Activation Function: ReLU
Batch Normalization: Applied after this layer
Convolutional Layer 4:
Number of Filters: 64
Filter Size: 3x3
Stride: 1
Padding: 1 (Same Padding)
Activation Function: ReLU
6. Second Pooling Layer
Type: Max Pooling
Pool Size: 2x2
Stride: 2
7. Second Dropout Layer
Dropout Rate: 0.25 (25% of neurons are dropped)
8. Flatten Layer
Purpose: Converts the 2D matrix to a 1D vector for the fully connected layer.
9. Fully Connected Layers
Fully Connected Layer 1:
Units: 512
Activation Function: ReLU
Batch Normalization: Applied after this layer
Dropout Rate: 0.25 (25% of neurons are dropped)
Fully Connected Layer 2:
Units: 1024
Activation Function: ReLU
Batch Normalization: Applied after this layer
Dropout Rate: 0.25 (25% of neurons are dropped)
Fully Connected Layer 3:
Units: 10 (One for each class in MNIST)
Activation Function: Softmax

10. Optimization Details
Optimizer: Adam
Loss Function: Categorical Crossentropy
Evaluation Metrics: Accuracy
Learning Rate: 0.001
Mini-Batch Size: 64
Epochs: 50
