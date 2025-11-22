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

# Drop rows with missing values (The dataset used has no missing values so this is not needed)
for row in df.index:
    if df.loc[row].isnull().sum() > 0:
        df = df.drop(row)

# Remove diabetes risk column as it may be a target variable
df_cleaned = df.drop("diabetes_risk_score", axis=1)
print(df_cleaned.head())

# Print unique values for non numerical columns
for col in df_cleaned.columns:
    if df_cleaned[col].dtype == 'object': # Categorical columns
        print(f"Unique values for {col}: {df_cleaned[col].unique()}")

# Create Graphics object from cleaned columns
graphics = Graphics(df_cleaned)

# Divide the columns into 3 quarters to show histograms and prevent overlapping
f_quarter_columns = df_cleaned.columns[1:4] # test column division
s_quarter_columns = df_cleaned.columns[4:7]
t_quarter_columns = df_cleaned.columns[7:10]

# Show histograms
graphics.show_histograms(f_quarter_columns)
graphics.show_histograms(s_quarter_columns)
graphics.show_histograms(t_quarter_columns)

# Convert object columns to numeric codes
# Encoding guide:
# Gender: Male = 0, Female = 1, Other = 2
# Ethnicity: Asian = 0, White = 1, Hispanic = 2, Black = 3, Other = 4
# Education level: Highschool = 0, Graduate = 1, Postgraduate = 2, No formal = 3
# Income level: Lower-Middle = 0, Middle = 1, Low = 2, Upper-Middle = 3, High = 4
# Employment status: Employed = 0, Unemployed = 1, Retired = 2, Student = 3
# Smoking status: Never = 0, Former = 1, Current = 2
# Diabetes stage: Type 2 = 0, No Diabetes = 1, Pre-Diabetes = 2, Gestational = 3, Type 1 = 4
#for col in df_cleaned.columns:
#    if df_cleaned[col].dtype == 'object':
#        df_cleaned[col] = df_cleaned[col].astype('category').cat.codes
