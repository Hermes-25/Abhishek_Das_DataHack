# Abhishek_Das_DataHack
Summer Analytics Hackathon 1




# 1. Data Loading and Merging:

It loads three separate datasets: one for training features, one for training labels, and one for test features.
It merges the training features (train_feat) with the training labels (train_label) on the common column 'respondent_id' to create a single dataset (train_data). This combined dataset will be used for training and validation.
# 2. Data Preparation:

Separation: After merging, it separates the dataset (train_data) into:
X: Contains all features except 'respondent_id', 'xyz_vaccine', and 'seasonal_vaccine'.
y_xyz: Target variable for predicting 'xyz_vaccine'.
y_seasonal: Target variable for predicting 'seasonal_vaccine'.
X_test: Contains features from the test dataset (test_feat) for making predictions.
# 3. Preprocessing Pipelines:

Numerical Pipeline: Uses SimpleImputer to handle missing values (filled with zeros) and StandardScaler to scale numerical features.
Categorical Pipeline: Uses SimpleImputer to handle missing values (filled with 'missing') and OneHotEncoder to encode categorical features.
Column Transformation:

Uses ColumnTransformer to apply different preprocessing pipelines to numerical and categorical columns separately:
Numerical columns are processed using the numerical pipeline (numerical_pipeline).
Categorical columns are processed using the categorical pipeline (categorical_pipeline).
# 4. Model Definition:

Defines two models:
model_xyz: Logistic Regression model for predicting 'xyz_vaccine'.
model_seasonal: Random Forest Classifier for predicting 'seasonal_vaccine'.
# 5. Pipeline Creation:

Creates separate pipelines (pipeline_xyz and pipeline_seasonal) for each target variable:
Each pipeline includes the respective preprocessing steps (preprocessor) followed by the classifier (model_xyz or model_seasonal).
# 6. Model Training and Evaluation:

Splitting Data: Splits X and y_xyz into training and validation sets (X_train_xyz, X_val_xyz, y_train_xyz, y_val_xyz) using train_test_split.
Training: Fits pipeline_xyz and pipeline_seasonal on the training data (X_train_xyz, X_train_seasonal) and their respective target variables (y_train_xyz, y_train_seasonal).
Prediction and Evaluation: Predicts probabilities (y_val_pred_prob_xyz, y_val_pred_prob_seasonal) for validation sets (X_val_xyz, X_val_seasonal) and calculates ROC AUC scores (roc_auc_xyz, roc_auc_seasonal) to evaluate model performance.
# 7. Model Deployment:

Retrains pipeline_xyz and pipeline_seasonal on the entire training data (X, y_xyz and X, y_seasonal, respectively).
Predicts probabilities for 'xyz_vaccine' and 'seasonal_vaccine' on the test dataset (X_test) and prepares a submission CSV (submission.csv) with 'respondent_id', 'xyz_vaccine', and 'seasonal_vaccine' columns.

# Summary
This script loads and merges datasets, prepares features and target variables, sets up preprocessing pipelines for numerical and categorical data, defines and trains machine learning models, evaluates model performance using ROC AUC scores, and finally makes predictions on new data for submission in a specific format required by the hackathon.
