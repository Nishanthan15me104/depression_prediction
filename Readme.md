
# Overview

**Kaggle competion - Exploring Mental Health Data**

**Datasets:** The dataset for this competition was generated from a deep learning model trained on the Depression Survey/Dataset for Analysis dataset. Feature distributions are close to, but not exactly the same, as the original.

**Goal:** goal is to use data from a mental health survey to explore factors that may cause individuals to experience depression.

# Tools I Used

For my deep dive into the data analyst job market, I harnessed the power of several key tools:

- **Python:** The backbone of my analysis, allowing me to analyze the data and find critical insights.I also used the following Python libraries:
    - **Pandas Library:** This was used to analyze the data. 
    - **Scikit-learn:**Leveraged  for streamlined ML workflows, encompassing data preprocessing, model training, hyperparameter tuning, and performance evaluation.
    - **Optuna:** Optimized Machine learning model hyperparameters with Optuna for enhanced classification performance in a streamlined Scikit-learn pipeline.
    - **Matplotlib Library:** I visualized the data.
    - **Seaborn Library:** Helped me create more advanced visuals. 
- **Jupyter Notebooks:** The tool I used to run my Python scripts which let me easily include my notes and analysis.
- **Visual Studio Code:** My go-to for executing my Python scripts.
- **Git & GitHub:** Essential for version control and sharing my Python code and analysis, ensuring collaboration and project tracking.

# Data Loading and Exploration

## Objective

- **Load training and testing datasets** using `pandas.read_csv()`.
- **Explore the data**:
  - Check for duplicates.
  - Describe basic statistics: number of rows, columns, data types, missing values, etc.

# Feature Engineering

## New Features Created

- **Age_Cut**: Categorizes age into bins using `pd.cut()`.
- **Name_Cut**: Categorizes names into bins using `pd.qcut()` (after removing duplicates).
- **CGPA_Cut**: Categorizes CGPA into bins (handles missing values for students).
- **Depression_Risk_Index**: Calculates a risk index based on CGPA and work/study hours (handles missing values).
- **Depression_Risk_Index_1** and **Depression_Risk_Index_2**: Categorize the risk index into bins using `pd.cut()` and `pd.qcut()`.
- **Study_Satisfaction_Ratio**: Ratio of study satisfaction and academic pressure.
- **Study_Satisfaction_Ratio_1**: Categorizes the above ratio into bins using `pd.qcut()`.
- **CGPA_Pressure_Ratio**: Ratio of CGPA and academic pressure.
- **CGPA_Pressure_Ratio1**: Categorizes the above ratio into bins using `pd.qcut()`.
- **Professional_Stress_Index**: Calculates stress index for working professionals.
- **Student_Stress_Index**: Calculates stress index for students (handles missing values).
- **Professional_Stress_Index2** and **Student_Stress_Index2**: Categorize stress indexes into bins using `pd.cut()`.

## Handling Missing Values

- **CGPA**: Missing CGPA filled with mean for students.
- **Job Satisfaction** and **Work Pressure**: Missing values filled with median for working professionals.
- **Study Satisfaction** and **Academic Pressure**: Missing values filled with median for students.

## Cleaning Categorical Features

- **Dietary Habits**: Missing values filled with mode; uncommon values grouped into 'Moderate'.
- **Financial Stress**: Missing values filled with mode.
- **Cleaned_Sleep_Duration**: Categories created for sleep duration based on ranges.
- **Cleaned_Profession**: Professions grouped into categories based on job type; missing values filled with 'Unemployed'.
- **Cleaned_Degree**: Degrees grouped based on the first letter; missing values filled with 'Others'.
- **City**: Missing values filled with the median depression mean for that city.

# Data Imputation and Encoding

## Techniques Used

- **Numerical Features**:
  - Mean/Median imputation.
  - Binning (using `pd.cut()` and `pd.qcut()`).
- **Categorical Features**:
  - Mode imputation.
  - Grouping uncommon values into broader categories.

  ## Data Preprocessing

### Splitting the Data
- `train_df.drop(['Depression'], axis=1)` creates a new DataFrame `X` containing all features except **Depression**.
- `train_df['Depression']` assigns the **Depression** column to `y` (target variable).
- `train_test_split` splits `X` and `y` into training and validation sets (`X_train`, `X_valid`, `y_train`, `y_valid`) for model training and evaluation.

## Feature Engineering with PCA
- Specific columns (**Pressure**, **Satisfaction**) are selected for dimensionality reduction using **PCA**.
- **StandardScaler** standardizes the data (important for PCA).
- **PCA** object is created to transform data into a lower-dimensional space (capturing most variance).
- Training and validation data are transformed using the fitted PCA model.
- New DataFrames (`pca_train_df`, `pca_valid_df`) are created with the PCA components.
- These DataFrames are optionally combined with the original features in `X_train` and `X_valid`.

## Preprocessing for Model Training

### Identifying Feature Types
- **Categorical features** are identified (`ohe_cols`, `ode_cols`).

### Feature Transformation Pipeline
- **ColumnTransformer** is used to combine different preprocessing steps for different feature types:
  - **SimpleImputer** imputes missing values with the most frequent value (`most_frequent`).
  - **OrdinalEncoder** handles ordinal features (e.g., dietary habits with levels like **good**, **bad**).
  - **OneHotEncoder** converts categorical features into one-hot encoded vectors.
  - `'passthrough'` keeps numerical features (e.g., **Pressure**, **Satisfaction**) unchanged.

### Training Transformation Pipeline
- The `col_trans` pipeline is fitted on the training data (`X_train`).

## Model Training with Hyperparameter Tuning - Best Models (Xgboost and Catboost)

### XGBoost Model and Catboost
- **XGBClassifier and Catboost** is chosen for classification (predicting depression risk).
- **GridSearchCV** could be used for hyperparameter tuning.

### Optuna for Hyperparameter Optimization
- **optuna** library is used for hyperparameter optimization of XGBoost:
  - `objective` function defines the hyperparameters (e.g., number of trees, learning rate) and evaluates the model using **accuracy** on the validation set.
  - `optuna.create_study` creates a study object to track optimization.
  - `study.optimize` runs the optimization process with a specified number of trials.
  - `best_params` retrieves the best hyperparameters found during optimization.

### Training the Final Model
- **XGBClassifier and Catboost** is instantiated with the best hyperparameters.
- `Pipelinexgb` combines the feature transformation pipeline (`col_trans`) with the XGBoost model.
- `Pipelinexgb.fit(X_train, y_train)` trains the final model on the entire training data.

## Model Evaluation and Prediction

### Performance on differtent  Dataset

#### Catboost
- the model's **accuracy** on the Validation set - 0.94083
- the model's **accuracy** on the Training set   - 0.94359
- the model's **accuracy** on the Public set     - 0.94253
- the model's **accuracy** on the Private set    - 0.94069

#### Xgboost
- the model's **accuracy** on the Validation set - 0.94033
- the model's **accuracy** on the Training set   - 0.94345
- the model's **accuracy** on the Public set     - 0.94211
- the model's **accuracy** on the Private set    - 0.94032


## Evaluation Metrics
- **confusion_matrix** shows the distribution of true vs. predicted labels.

** Best model - Catboost** 
[22283   744]
[  921  4192]

- **classification_report** provides various performance metrics like **precision**, **recall**, **F1-score**, and **support** for each class (**depression**, **no depression**).

**Best model - Catboost**

precision    recall  f1-score   support

           0       0.96      0.97      0.96     
           1       0.85      0.82      0.83    

    accuracy                           0.94     
    macro avg       0.90      0.89     0.90     
    weighted avg    0.94      0.94     0.94     

