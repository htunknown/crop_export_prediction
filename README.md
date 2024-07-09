# Machine Learning Model for Predicting Consumer Prices
Description

This project focuses on developing a machine learning model to predict consumer prices using various economic indicators. The aim is to create a reliable model that can forecast future prices based on historical data.

Features and Labels
The dataset includes several features such as country area, year, and consumer price indicators. Unnecessary columns like "Domain Code," "Area Code," "Year Code," "Item Code," "Months Code," "Element Code," "Flag," "Flag Description," and "Note" were excluded. Exchange rate values were also excluded to ensure a more accurate analysis ​​.

Key Features:
Area: Common column across all datasets.
Year: Data filtered from 2010 to 2019.
Consumer Price Index (CPI): Adjusted to 2015 as the base year​​.
Preprocessing
Missing Values
Interpolation: Linear interpolation was used for handling missing values before and after merging datasets. Null values remaining after interpolation led to dropping specific areas to maintain data integrity​​.
Encoding Categorical Variables
Label Encoding: Applied to the "Area" column which contains country names​​.
Model Architecture
MLP Model
The model is a Multi-Layer Perceptron (MLP) consisting of:

Input Layer: Determined by the number of features in the dataset.
Hidden Layers:
First hidden layer with 64 units using ReLU activation.
Second hidden layer with 32 units using ReLU activation.
Dropout layer with a rate of 0.1 to prevent overfitting​​.
Output Layer: Single unit with linear activation​​.
Loss Function
Mean Squared Error (MSE): Used to train the model​​.
Preventing Overfitting
Dropout Layer: Randomly drops 10% of the neurons during training.
EarlyStopping: Monitors validation loss with a patience of 10 epochs.
Validation Split: 20% of data used for validation to evaluate model performance on unseen data​​.
Performance Metrics
Metrics used to measure the performance include:

Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)​​.
Results
Before fine-tuning, the model had the following performance:

Mean Squared Error: 1.7263661445643807e+18
Root Mean Squared Error: 1313912533.0722668
Mean Absolute Error: 597358726.2049419
After experimenting with different hyperparameters, the final model was optimized for better performance​​.
