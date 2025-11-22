import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil
import kagglehub
from Graphics import Graphics

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
print(df.isnull().sum())

# Clean the data
exclusions = "diabetes_risk_score"
df_cleaned = df.drop(exclusions, axis=1)
print(df_cleaned.head())

# Print unique values for non numerical columns
for col in df_cleaned.columns:
    if df_cleaned[col].dtype == 'object': # Categorical columns
        print(f"Unique values for {col}: {df_cleaned[col].unique()}")

# Convert object columns to numeric codes
for col in df_cleaned.columns:
    if df_cleaned[col].dtype == 'object':
        df_cleaned[col] = df_cleaned[col].astype('category').cat.codes


# Create Graphics object from cleaned columns
graphics = Graphics(df_cleaned)

# Divide the columns into 3 quarters to show histograms and prevent overlapping
f_quarter_columns = df_cleaned.columns[1:4] # test column division
# Show histograms
graphics.show_histograms(f_quarter_columns)
