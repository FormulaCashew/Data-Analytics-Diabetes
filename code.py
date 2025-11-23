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
if df.isnull().sum().sum() > 0:
    print(df.isnull().sum())
    drop_rows = True
else:
    print("No missing values in the dataset")
    drop_rows = False

# Drop rows with missing values if necessary
if drop_rows: 
    for row in df.index:
        if df.loc[row].isnull().sum() > 0:
            df = df.drop(row)

# Remove diabetes risk column as it may be a target variable
df_cleaned = df.drop("diabetes_risk_score", axis=1)
print("Cleaned dataframe:", df_cleaned.head())

# Count the number of rows and columns
print("Dataframe shape:", df_cleaned.shape)

# Get numerical columns
num_columns = df_cleaned.select_dtypes(include=[np.number]).columns
print("Numerical columns:", num_columns)

# Get categorical columns
cat_columns = df_cleaned.select_dtypes(include= 'object').columns
print("Categorical columns:", cat_columns)

# Print unique values for non numerical columns
for col in df_cleaned.columns:
    if df_cleaned[col].dtype == 'object': # Categorical columns
        print(f"Unique values for {col}: {df_cleaned[col].unique()}")

################################ EDA Visualization #################################

# Create Graphics object from cleaned columns
processor = DataProcessor(df_cleaned)
df_subsampled = processor.subsample_data(fraction=0.05)
graphics = Graphics(df_subsampled)

important_attributes = ["age", "bmi", "glucose_fasting", "hba1c", "cholesterol_total"]

# Get numerical columns and divide them into groups of 6
first_six_cols = df_subsampled.select_dtypes(include= np.number).columns[:6]
second_six_cols = df_subsampled.select_dtypes(include= np.number).columns[6:12]
third_six_cols = df_subsampled.select_dtypes(include= np.number).columns[12:18]
fourth_six_cols = df_subsampled.select_dtypes(include= np.number).columns[18:24]
fifth_six_cols = df_subsampled.select_dtypes(include= np.number).columns[24:30]
important_columns = df_subsampled[important_attributes].columns

# Show histograms
if False : # Toggle to show histograms
    graphics.show_histograms(first_six_cols)
    graphics.show_histograms(second_six_cols)
    graphics.show_histograms(third_six_cols)
    graphics.show_histograms(fourth_six_cols)
    graphics.show_histograms(fifth_six_cols)
graphics.show_histograms(important_columns)

# Show correlation matrix
if False:
    graphics.show_correlation_matrix(num_columns)
graphics.show_correlation_matrix(important_columns)

# Show scatter matrix
if False:
    graphics.show_scatter_matrix(num_columns[:5])
graphics.show_scatter_matrix(important_columns)

# Show boxplots to check for outliers
if True:
    graphics.show_boxplots(important_columns)
    

################################ Normalization ################################

# Normalize numerical data
processor = DataProcessor(df_cleaned)
norm_df = processor.normalize_data(df_cleaned.select_dtypes(include=[np.number]))

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
for col in norm_df.columns:
    if norm_df[col].dtype == 'object':
        norm_df[col] = norm_df[col].astype('category').cat.codes

# Print dtypes to check if encoding was successful
print(norm_df.dtypes)

# Update objects from class with cleaned columns
df_subsampled_num_encoded = processor.subsample_data(fraction=0.05)
data_processor = DataProcessor(df_subsampled_num_encoded)

# Show kmeans clustering
# data_processor.plot_kmeans_clustering(["cholesterol_total","diagnosed_diabetes"])
