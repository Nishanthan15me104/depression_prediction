# Data Loading and Exploration

## Objective

- **Load training and testing datasets** using `pandas.read_csv()`.
- **Explore the data**:
  - Check for duplicates.
  - Describe basic statistics: number of rows, columns, data types, missing values, etc.

## Tools and Libraries

- **Python**: Used for data manipulation and exploration.
- **Libraries**:
  - **Pandas**: For data loading and processing.
  - **NumPy**: For numerical computations.

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

## Model Training with Hyperparameter Tuning

### XGBoost Model
- **XGBClassifier** is chosen for classification (predicting depression risk).
- **GridSearchCV** could be used for hyperparameter tuning (commented out).

### Optuna for Hyperparameter Optimization
- **optuna** library is used for hyperparameter optimization of XGBoost:
  - `objective` function defines the hyperparameters (e.g., number of trees, learning rate) and evaluates the model using **accuracy** on the validation set.
  - `optuna.create_study` creates a study object to track optimization.
  - `study.optimize` runs the optimization process with a specified number of trials.
  - `best_params` retrieves the best hyperparameters found during optimization.

### Training the Final Model
- **XGBClassifier** is instantiated with the best hyperparameters.
- `Pipelinexgb` combines the feature transformation pipeline (`col_trans`) with the XGBoost model.
- `Pipelinexgb.fit(X_train, y_train)` trains the final model on the entire training data.

## Model Evaluation and Prediction

### Performance on Validation Set
- `Pipelinexgb.score(X_valid, y_valid)` evaluates the model's **accuracy** on the validation set.
- `Pipelinexgb.score(X_train, y_train)` evaluates the model's **accuracy** on the training set (may lead to overfitting).

### Prediction on Test Set
- `Pipelinexgb.predict(X_valid)` predicts depression risk for the validation set.
- `Pipelinexgb.predict(test_df)` predicts depression risk for the unseen test set.

## Generating Submission File
- `submission1` DataFrame stores predictions for the test set.
- `submission1.to_csv` saves the predictions to a CSV file.

## Evaluation Metrics
- **confusion_matrix** shows the distribution of true vs. predicted labels.
- **classification_report** provides various performance metrics like **precision**, **recall**, **F1-score**, and **support** for each class (**depression**, **no depression**).

