import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil
import kagglehub

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
