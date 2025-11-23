import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil
import kagglehub
from Class_implementations import Graphics, DataProcessor


################################ Data Loading ################################
csv_name = "diabetes_dataset.csv"

# Download dataset if it doesn't exist
if (not os.path.exists(csv_name)):
    path = kagglehub.dataset_download("mohankrishnathalla/diabetes-health-indicators-dataset")
    print("Path to dataset files:", path)

    # move dataset to local directory
    try:
        path = shutil.move(path+"/"+csv_name, ".")
        print("New path to dataset files:", path)
    except:
        print("Error moving dataset files")
else:
    print("Dataset already exists in current directory")

# Read the data
df = pd.read_csv(csv_name)
 

################################ Initial Data Cleaning ################################

# Print the first 5 rows of the DataFrame
print(df.head())

# Calculate the number of missing values in each column
missing_values = df.isnull().sum().sum()
if missing_values > 0:
    print(f"{missing_values} missing values in the dataset")
    # Drop rows with missing values if necessary
    df_dropped = df.dropna()
else:
    print("No missing values in the dataset")
    df_dropped = df.copy()

# Remove diabetes risk column as it may be a target variable
df_dropped = df_dropped.drop("diabetes_risk_score", axis=1)
print("Cleaned dataframe:", df_dropped.head())

# Count the number of rows and columns
print("Dataframe shape:", df_dropped.shape)

# Get numerical columns
num_columns = df_dropped.select_dtypes(include=[np.number]).columns
print("Numerical columns:", num_columns)

# Get categorical columns
cat_columns = df_dropped.select_dtypes(include= 'object').columns
print("Categorical columns:", cat_columns)

# Print unique values for non numerical columns
for col in df_dropped.columns:
    if df_dropped[col].dtype == 'object': # Categorical columns
        print(f"Unique values for {col}: {df_dropped[col].unique()}")

################################ EDA Visualization #################################

# Create Graphics object from cleaned columns
processor = DataProcessor(df_dropped)
df_subsampled = processor.subsample_data(fraction=0.1)
graphics = Graphics(df_subsampled)

# Show correlation matrix
if False:
    graphics_total = Graphics(df_dropped) # Important to use the total dataset to show correlation matrix
    graphics_total.show_correlation_matrix(num_columns)

# Based on correlation matrix above, select important numerical attributes with correlation > 0.1
important_attributes = ["age", "family_history_diabetes", "bmi", "glucose_fasting", "glucose_postprandial", "hba1c", "systolic_bp"]
important_attributes_w_target = important_attributes + ["diagnosed_diabetes"]

important_columns = df_subsampled[important_attributes].columns
important_columns_w_target = df_subsampled[important_attributes_w_target].columns

# See distribution of important attributes
graphics.show_histograms(important_columns)
# Confirm correlation matrix values
graphics.show_correlation_matrix(important_columns_w_target)

# Show scatter matrix
if False:
    graphics.show_scatter_matrix(important_columns_w_target)

# Show boxplots to check for outliers
if False:
    graphics.show_boxplots(important_columns)
    

################################ Normalization ################################

# Normalize numerical data
processor = DataProcessor(df_dropped)
norm_df = processor.normalize_data(df_dropped.select_dtypes(include=[np.number]))

################################ Encoding #####################################

# Convert object columns to numeric codes
# Encoding guide:
# Gender: Male = 0, Female = 1, Other = 2
# Ethnicity: Asian = 0, White = 1, Hispanic = 2, Black = 3, Other = 4
# Education level: Highschool = 0, Graduate = 1, Postgraduate = 2, No formal = 3
# Income level: Lower-Middle = 0, Middle = 1, Low = 2, Upper-Middle = 3, High = 4
# Employment status: Employed = 0, Unemployed = 1, Retired = 2, Student = 3
# Smoking status: Never = 0, Former = 1, Current = 2
# Diabetes stage: Type 2 = 0, No Diabetes = 1, Pre-Diabetes = 2, Gestational = 3, Type 1 = 4
encoded_df = norm_df.copy()
for col in norm_df.columns:
    if norm_df[col].dtype == 'object':
        encoded_df[col] = norm_df[col].astype('category').cat.codes

# Print dtypes to check if encoding was successful
print(encoded_df.dtypes)

# Update classes
processor = DataProcessor(encoded_df)
graphics = Graphics(encoded_df) # Update Graphics object with encoded data

# check for correlation matrix values now for categorical data
cols_to_check = cat_columns.tolist() + ['diagnosed_diabetes']
print("cols_to_check:", cols_to_check)
graphics.show_correlation_matrix(cols_to_check)
# After encoding, correlation matrix values show little correlation with target variable

# Update objects from class with cleaned columns
df_subsampled_num_encoded = processor.subsample_data(fraction=0.05)
data_processor = DataProcessor(df_subsampled_num_encoded)

# Show kmeans clustering
# data_processor.plot_kmeans_clustering(["cholesterol_total","diagnosed_diabetes"])
