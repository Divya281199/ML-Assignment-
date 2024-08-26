For Successful execution of this code we will required one csv named heart.csv have attached it on mainbranch for your readyreference.
Steps Explained:
Loading the Dataset: The dataset is loaded from the UCI repository with predefined column names.
Data Preprocessing:
Missing values represented by '?' are replaced with NaN.
Convert all columns to numeric data types.
Drop any rows that contain missing values.
The target variable is adjusted so that 0 indicates no heart disease, while any other value (1-4) indicates the presence of heart disease.
Splitting the Dataset: The dataset is split into training and testing sets.
Building and Training the Model: A Random Forest Classifier is trained on the dataset.
Evaluating the Model:
Accuracy: Measures the percentage of correct predictions.
Confusion Matrix: Shows the breakdown of true positives, true negatives, false positives, and false negatives.
Classification Report: Provides precision, recall, f1-score, and support for each class.
Visualizing the Confusion Matrix: A heatmap of the confusion matrix is plotted using Seaborn.
Feature Importance (Optional): The importance of each feature used by the model is plotted, providing insights into which features are most influential in predicting heart disease.
