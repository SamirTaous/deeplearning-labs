

# **Lab 1**

## Regression & Multi-Class Classification

## Report

### Objective
This lab aimed to explore deep learning models using PyTorch to address two types of machine learning problems: **regression** and **multi-class classification**. The goal was to establish, optimize, and evaluate deep neural networks (DNNs) for both tasks while applying various techniques such as hyperparameter tuning, regularization, and model evaluation.

### Part One: Regression (NYSE Dataset)
#### Task:
For the regression task, I worked with the [NYSE dataset](https://www.kaggle.com/datasets/dgawlik/nyse) to predict stock prices. The process involved building a deep neural network (DNN) architecture to handle regression, hyperparameter tuning with GridSearch from the sklearn library, and applying regularization techniques.

#### Key Steps:
1. **Exploratory Data Analysis (EDA)**: 
   - I visualized the dataset to understand its structure and identify key features that might influence stock prices.
2. **Model Development**:
   - A deep neural network was designed using PyTorch for regression tasks.
   - Various architectures were experimented with, adjusting the number of layers and neurons per layer.
3. **Hyperparameter Tuning**:
   - GridSearch was applied to find the best hyperparameters, including learning rate, optimizer choice, and the number of epochs.
4. **Regularization**:
   - Dropout and weight decay were used to prevent overfitting and improve generalization.
5. **Model Evaluation**:
   - The model's performance was visualized by plotting the loss and accuracy curves over epochs.
   - Metrics like Mean Squared Error (MSE) were used to evaluate model performance.

#### Results:
After training the model with optimal hyperparameters, the regression model showed improved performance, with significant reductions in loss and more stable predictions on test data.

### Part Two: Multi-Class Classification (Predictive Maintenance Dataset)
#### Task:
For the classification task, I worked with the [Predictive Maintenance dataset](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification) to predict the condition of machinery based on sensor data. The task involved preprocessing the dataset, balancing the classes, and establishing a DNN architecture for multi-class classification.

#### Key Steps:
1. **Data Preprocessing**:
   - The data was cleaned, standardized, and normalized to prepare it for the model.
2. **Data Augmentation**:
   - Techniques like oversampling were used to balance the classes and avoid the model becoming biased toward the majority class.
3. **Exploratory Data Analysis (EDA)**:
   - Visualizations helped to better understand the distribution of different features in the dataset.
4. **Model Development**:
   - A deep neural network was created to classify the data into multiple classes, experimenting with different architectures and layer configurations.
5. **Hyperparameter Tuning**:
   - GridSearch was employed to identify the best set of hyperparameters, such as the learning rate, optimizer, and number of epochs.
6. **Regularization**:
   - Dropout and L2 regularization were applied to prevent overfitting.
7. **Model Evaluation**:
   - The model was evaluated using classification metrics like accuracy, sensitivity, and F1-score.

#### Results:
The multi-class classification model showed strong performance after tuning. The final model achieved high accuracy and balanced sensitivity across different classes. The use of data augmentation significantly improved model generalization.

### Conclusion
This lab was a hands-on exercise in building deep learning models for both regression and classification tasks using PyTorch. I gained valuable experience in preprocessing data, designing and tuning deep neural networks, and evaluating model performance with various metrics. Applying regularization techniques helped improve model generalization and mitigate overfitting, leading to more robust models.

This lab reinforced my understanding of:
- Data preprocessing and cleaning
- Hyperparameter tuning and GridSearch
- Regularization techniques to improve model robustness
- Evaluating models using appropriate metrics

These techniques are crucial for building reliable machine learning models in real-world applications.

# **Lab 2**

## CNN and Faster R-CNN for MNIST

## Report

### Objective

This lab explored different neural network architectures for image classification and object detection, focusing on the MNIST dataset. The goal was to implement, train, evaluate, and compare a Convolutional Neural Network (CNN) and a Faster R-CNN model for the task of digit recognition. The lab investigated the suitability of an object detection model (Faster R-CNN) for a simpler classification problem and highlighted the trade-offs between model complexity and performance.

### Part One: CNN Classifier

#### Task:

For the first part, a Convolutional Neural Network (CNN) was designed and trained to classify the MNIST digits. This involved defining the network architecture, setting hyperparameters, implementing the training loop, and evaluating the model's performance on the test set.

#### Key Steps:

1. **Data Loading and Preprocessing:**
    
    - The MNIST dataset was loaded from ubyte files using a custom data loading function (read_idx).
        
    - A custom Dataset class (MNISTUByteDataset) was created to handle the ubyte data format and normalize the pixel values.
        
    - DataLoader instances were created for training and testing, enabling batching and shuffling.
        
2. **CNN Model Definition:**
    
    - A CNN architecture was defined using PyTorch's nn.Module. The architecture included convolutional layers, max pooling layers, and fully connected layers.
        
    - ReLU activation functions were used after the convolutional and fully connected layers.
        
3. **Training Setup:**
    
    - The CrossEntropyLoss was used as the loss function.
        
    - The Adam optimizer was used with a learning rate of 0.001.
        
    - The training loop iterated over epochs and batches, performing the forward pass, calculating the loss, performing backpropagation, and updating the model parameters.
        
    - Training loss and validation accuracy were tracked during training.
        
4. **Evaluation:**
    
    - The trained CNN model was evaluated on the test set.
        
    - Metrics such as accuracy, F1-score, and a confusion matrix were calculated and reported.
        
    - The training loss and validation accuracy were plotted over epochs to analyze the training process.
        

#### Results:

The CNN model achieved a test accuracy of 98.83%, demonstrating strong performance on the MNIST classification task. The training loss converged smoothly, and the validation accuracy reached a plateau, indicating that the model was learning effectively without significant overfitting. The training time for this model was relatively small.

### Part Two: Faster R-CNN for MNIST

#### Task:

In the second part, the MNIST dataset was adapted for use with a Faster R-CNN model, treating each digit as an object to be detected. A pre-trained Faster R-CNN model was fine-tuned for this task, and its performance was compared to the CNN model.

#### Key Steps:

1. **Data Transformation for Object Detection:**
    
    - A custom Dataset class (MNISTObjectDetectionDataset) was created to transform the MNIST data into an object detection format.
        
    - Each digit was treated as an object with a bounding box covering most of the image.
        
    - The dataset class created target dictionaries containing the bounding box coordinates and labels.
        
2. **Faster R-CNN Model Setup:**
    
    - A pre-trained Faster R-CNN model with a ResNet50 backbone (fasterrcnn_resnet50_fpn) was loaded from torchvision.models.detection.
        
    - The classifier (box predictor) of the Faster R-CNN model was modified to match the number of classes in the MNIST dataset (10 digits + background).
        
3. **Training:**
    
    - The training loop iterated over epochs and batches, performing the forward pass, calculating the loss, and updating the model parameters.
        
    - The Adam optimizer was used with a lower learning rate (0.0005) due to fine-tuning a pre-trained model.
        
4. **Evaluation:**
    
    - The trained Faster R-CNN model was evaluated on the test set.
        
    - Metrics such as accuracy, F1-score, and a confusion matrix were calculated and reported. Since Faster R-CNN gives bounding boxes, the overlap of the prediction with the actual bounding box was be taken into account.
        

#### Results:

The Faster R-CNN model achieved a test accuracy of 95.23 %. While it performed reasonably well, the training time was significantly longer than the CNN model. The more complex architecture of Faster R-CNN did not result in a significant improvement in accuracy for this relatively simple digit classification task.

### Conclusion

This lab provided valuable insights into the application of different neural network architectures for image classification and object detection. While Faster R-CNN is a powerful model for complex object detection tasks, it proved to be overkill for the MNIST dataset. The simpler CNN architecture achieved comparable or better accuracy with significantly less training time, making it a more efficient and suitable solution for this specific problem.  
Key observations included:

- the CNN performed very well
    
- The Faster R-CNN required more setup and time to implement
    
- The Faster R-CNN is not an elegant way to deal with image classification
