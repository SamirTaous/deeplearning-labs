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