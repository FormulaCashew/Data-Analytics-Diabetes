import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil
import kagglehub
from Class_implementations import Graphics, DataProcessor

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
 
# Print the first 5 rows of the DataFrame
print(df.head())

# Calculate the number of missing values in each column
if df.isnull().sum().sum() > 0:
    print(df.isnull().sum())
else:
    print("No missing values in the dataset")

# Drop rows with missing values (The dataset used has no missing values so this is not needed)
for row in df.index:
    if df.loc[row].isnull().sum() > 0:
        df = df.drop(row)

# Remove diabetes risk column as it may be a target variable
df_cleaned = df.drop("diabetes_risk_score", axis=1)
print(df_cleaned.head())

# Count the number of rows and columns
print("Dataframe shape:", df.shape)

# Get numerical columns
num_columns = df.select_dtypes(include=[np.number]).columns
print("Numerical columns:", num_columns)

# Get categorical columns
cat_columns = df.select_dtypes(include= 'object').columns
print("Categorical columns:", cat_columns)


# Print unique values for non numerical columns
for col in df_cleaned.columns:
    if df_cleaned[col].dtype == 'object': # Categorical columns
        print(f"Unique values for {col}: {df_cleaned[col].unique()}")

# Create Graphics object from cleaned columns
processor = DataProcessor(df_cleaned)
df_subsampled = processor.subsample_data(fraction=0.05)
graphics = Graphics(df_subsampled)

# Divide the columns into 3 quarters to show histograms and prevent overlapping
first_six_cols = df_subsampled.columns[:6]
second_six_cols = df_subsampled.columns[6:12]
third_six_cols = df_subsampled.columns[12:18]
fourth_six_cols = df_subsampled.columns[18:24]
fifth_six_cols = df_subsampled.columns[24:30]

# Show histograms
if True : # Toggle to show histograms
    graphics.show_histograms(first_six_cols)
    graphics.show_histograms(second_six_cols)
    graphics.show_histograms(third_six_cols)
    graphics.show_histograms(fourth_six_cols)
    graphics.show_histograms(fifth_six_cols)

########################### Encoding ###########################

# Convert object columns to numeric codes
# Encoding guide:
# Gender: Male = 0, Female = 1, Other = 2
# Ethnicity: Asian = 0, White = 1, Hispanic = 2, Black = 3, Other = 4
# Education level: Highschool = 0, Graduate = 1, Postgraduate = 2, No formal = 3
# Income level: Lower-Middle = 0, Middle = 1, Low = 2, Upper-Middle = 3, High = 4
# Employment status: Employed = 0, Unemployed = 1, Retired = 2, Student = 3
# Smoking status: Never = 0, Former = 1, Current = 2
# Diabetes stage: Type 2 = 0, No Diabetes = 1, Pre-Diabetes = 2, Gestational = 3, Type 1 = 4
for col in df_cleaned.columns:
    if df_cleaned[col].dtype == 'object':
        df_cleaned[col] = df_cleaned[col].astype('category').cat.codes

# Print dtypes to check if encoding was successful
print(df_cleaned.dtypes)

# Divide again columns as they need to be updated after encoding
first_six_cols = df_cleaned.columns[:6]
second_six_cols = df_cleaned.columns[6:12]
third_six_cols = df_cleaned.columns[12:18]
fourth_six_cols = df_cleaned.columns[18:24]
fifth_six_cols = df_cleaned.columns[24:30]

# Create new Graphics object from cleaned columns
df_subsampled_num_encoded = processor.subsample_data(fraction=0.1)
graphics_num_encoded = Graphics(df_subsampled_num_encoded)

# Show correlation matrix
if False: # Toggle to show correlation matrix
    graphics_num_encoded.show_correlation_matrix(first_six_cols)
    graphics_num_encoded.show_correlation_matrix(second_six_cols)
    graphics_num_encoded.show_correlation_matrix(third_six_cols)
    graphics_num_encoded.show_correlation_matrix(fourth_six_cols)
    graphics_num_encoded.show_correlation_matrix(fifth_six_cols)
    graphics_num_encoded.show_correlation_matrix_all()

# Scatter matrix with certain columns
graphics_num_encoded.show_scatter_matrix(["bmi", "diagnosed_diabetes", "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides"])

