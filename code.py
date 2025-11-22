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

# Print histograms
graphics = Graphics(df_cleaned)
graphics.show_histograms()
