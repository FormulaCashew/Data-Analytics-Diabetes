import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import shutil
import kagglehub
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from Class_implementations import Graphics, DataProcessor, KNN


################################ Data Loading ################################
csv_name = "diabetes_dataset.csv"

# Download dataset if it doesn't exist
if (not os.path.exists(csv_name)):
    path = kagglehub.dataset_download("mohankrishnathalla/diabetes-health-indicators-dataset")
    print("Path to dataset files:", path)

    # move dataset to local directory
    try:
        path = shutil.move(os.path.join(path, csv_name), ".")
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

# Create DataProcessor object from cleaned columns
processor = DataProcessor(df_dropped)

# Remove outliers from numerical columns which are not binary
cols_outlier_check = ["age", "alcohol_consumption_per_week", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day", "hypertension_history", "cardiovascular_history", "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides", "glucose_fasting", "glucose_postprandial", "insulin_level", "hba1c"]
processor.remove_outliers(cols_outlier_check, threshold=2.0)
df_dropped = processor.get_data() # Update dataframe using a copy
print("Dataframe shape after removing outliers:", df_dropped.shape)

# Subsample data
df_subsampled = processor.subsample_data(fraction=0.1)
graphics = Graphics(df_subsampled)

# Show correlation matrix
if False:
    graphics_total = Graphics(df_dropped) # Important to use the total dataset to show correlation matrix
    graphics_total.show_correlation_matrix(num_columns)

# Based on correlation matrix above, select important numerical attributes with correlation > 0.1, ordered by correlation
important_attributes = ["hba1c", "glucose_postprandial", "glucose_fasting", "family_history_diabetes", "age", "bmi", "systolic_bp"]
important_attributes_w_target = important_attributes + ["diagnosed_diabetes"]

important_columns = df_subsampled[important_attributes].columns
important_columns_w_target = df_subsampled[important_attributes_w_target].columns

# See distribution of important attributes
graphics.show_histograms(important_columns)
# Confirm correlation matrix values
graphics.show_correlation_matrix(important_columns_w_target)

# Show scatter matrix
if True:
    graphics.show_scatter_matrix(important_columns_w_target)

# Show boxplots to check for outliers
if True:
    graphics.show_boxplots(important_columns)
    

################################ Normalization ################################

# Normalize numerical data
processor = DataProcessor(df_dropped)
cols_to_norm = df_dropped.select_dtypes(include=[np.number]).columns.tolist()
if "diagnosed_diabetes" in cols_to_norm:
    cols_to_norm.remove("diagnosed_diabetes")
norm_df = processor.normalize_data(cols_to_norm)

# Check if done successfully
print(norm_df['age'].head())

################################ Encoding #####################################

# Drop rows with gender = Other as it not an useful feature
norm_df = norm_df[norm_df['gender'] != 'Other']

# Convert object columns to numeric codes
# Encoding guide:
# Gender: Male = 0, Female = 1
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

################################ EDA Visualization Part 2 ################################

# Update classes
processor = DataProcessor(encoded_df)
graphics = Graphics(encoded_df) # Update Graphics object with encoded data

# check for correlation matrix values now for categorical data
cols_to_check = cat_columns.tolist() + ['diagnosed_diabetes']
print("cols_to_check:", cols_to_check)
graphics.show_correlation_matrix(cols_to_check)
# After encoding, correlation matrix values show little correlation with target variable

graphics.compare_hist("glucose_fasting", "glucose_postprandial")
################# Histograms #################
# Histogram of education level vs diagnosed_diabetes
graphics.show_hist_axis("education_level", "diagnosed_diabetes")
# It shows that lower education level has higher count of diagnosed diabetes

# Histogram of income level vs diagnosed_diabetes
graphics.show_hist_axis("income_level", "diagnosed_diabetes")
# Shows that higher income level has higher count of diagnosed diabetes, but may be due to having the chance to be diagnosed

# Histogram of employment status vs diagnosed_diabetes
graphics.show_hist_axis("employment_status", "diagnosed_diabetes")
# Shows that employed has higher count of diagnosed diabetes

# Histogram of smoking status vs diagnosed_diabetes
graphics.show_hist_axis("smoking_status", "diagnosed_diabetes")
# Shows that current smoking has higher count of diagnosed diabetes

important_attributes = ["hba1c", "glucose_postprandial", "glucose_fasting", "family_history_diabetes", "age", "bmi", "systolic_bp", "smoking_status", "employment_status", "education_level", "income_level"]
important_attributes_w_target = important_attributes + ["diagnosed_diabetes"]

################# KMeans Clustering #################
# Update objects from class with cleaned columns
df_subsampled_num_encoded = processor.subsample_data(fraction=0.1)
data_processor = DataProcessor(df_subsampled_num_encoded)

# Show kmeans clustering
# Using hba1c and glucose fasting as they have the highest correlation with target variable
data_processor.plot_kmeans_clustering(["hba1c","glucose_fasting"], n_clusters=2)
# Now test for hba1c and glucose_postprandial
data_processor.plot_kmeans_clustering(["hba1c","glucose_postprandial"], n_clusters=2)
# Now test for glucose fasting and glucose_postprandial
data_processor.plot_kmeans_clustering(["glucose_fasting","glucose_postprandial"], n_clusters=2)


####################################### Modeling #########################################

# IMPORTANT: Use 5% to 10% of the dataset for training and testing, knn is heavy and takes a lot of time to run
subsampled_df = processor.subsample_data(fraction=0.05)
subsampled_processor = DataProcessor(subsampled_df)
train_df, test_df = subsampled_processor.train_test_split(test_size=0.2)
print(train_df.shape)
print(test_df.shape)

################# Decision Tree Library #################

if True:
    # Create Decision Tree object
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(train_df[important_attributes], train_df['diagnosed_diabetes'])

    # Make predictions
    predictions = decision_tree.predict(test_df[important_attributes])
    # Calculate accuracy
    correct = sum(predictions == test_df['diagnosed_diabetes'])
    print(f"Accuracy for Decision Tree: {correct/len(test_df)}")
    decision_tree_accuracy = correct/len(test_df)
    # Plot confusion matrix
    cm = confusion_matrix(test_df['diagnosed_diabetes'], predictions)
    sns.heatmap(cm, annot=True)
    plt.title("Confusion Matrix Decision Tree")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    recall_decision_tree = cm[1][1] / (cm[1][1] + cm[1][0])
    print(f"Recall for Decision Tree: {recall_decision_tree}")

################# KNN Library #################

if True:
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(train_df[important_attributes], train_df['diagnosed_diabetes'])
    # Make predictions
    predictions = knn.predict(test_df[important_attributes])
    # Calculate accuracy
    correct = sum(predictions == test_df['diagnosed_diabetes'])
    print(f"Accuracy for KNN: {correct/len(test_df)}")
    knn_accuracy = correct/len(test_df)
    # Plot confusion matrix
    cm = confusion_matrix(test_df['diagnosed_diabetes'], predictions)
    sns.heatmap(cm, annot=True)
    plt.title("Confusion Matrix KNN Library")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print(cm)

    recall_knn = cm[1][1] / (cm[1][1] + cm[1][0])
    print(f"Recall for KNN: {recall_knn}")

################# KNN Personal #################
if False:
    # This code has been disabled as it takes a lot of time to run
    # During testing, it was found to give the same results as the library implementation
    # It is left here for reference, but the library implementation is preferred due to its speed
    # Create KNN object
    knn = KNN(k=3)
    knn.store(train_df[important_attributes], train_df['diagnosed_diabetes'])
    # Make predictions
    predictions = knn.predict(test_df[important_attributes])
    # Calculate accuracy
    correct = sum(predictions == test_df['diagnosed_diabetes'])
    print(f"Accuracy: {correct/len(test_df)}")
    # Plot confusion matrix
    knn.plot_confusion_matrix(test_df[important_attributes], test_df['diagnosed_diabetes'])


################# Random Forest #################
if True:
    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(train_df[important_attributes], train_df['diagnosed_diabetes'])
    # Make predictions
    predictions = random_forest.predict(test_df[important_attributes])
    # Calculate accuracy
    correct = sum(predictions == test_df['diagnosed_diabetes'])
    print(f"Accuracy for Random Forest: {correct/len(test_df)}")
    random_forest_accuracy = correct/len(test_df)
    # Plot confusion matrix
    cm = confusion_matrix(test_df['diagnosed_diabetes'], predictions)
    sns.heatmap(cm, annot=True)
    plt.title("Confusion Matrix Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print(cm)

    # Plot feature importance
    feature_importance = random_forest.feature_importances_
    feature_names = train_df[important_attributes].columns
    feature_importance = pd.Series(feature_importance, index=feature_names)
    feature_importance.sort_values(ascending=False, inplace=True)
    feature_importance.plot(kind='bar')
    plt.title("Feature Importance Random Forest")
    plt.show()
    # Shows a bar chart a bit different than the one from decision tree
    # Glucose postprandial has a higher importance but hba1c is still the most important
    
    recall_random_forest = cm[1][1] / (cm[1][1] + cm[1][0])
    print(f"Recall for Random Forest: {recall_random_forest}")
###################### XG Boost ####################

if True:
    xgboost = XGBClassifier()
    xgboost.fit(train_df[important_attributes], train_df['diagnosed_diabetes'])
    # Make predictions
    predictions = xgboost.predict(test_df[important_attributes])
    # Calculate accuracy
    correct = sum(predictions == test_df['diagnosed_diabetes'])
    print(f"Accuracy for XG Boost: {correct/len(test_df)}")
    xgboost_accuracy = correct/len(test_df)
    # Plot confusion matrix
    cm = confusion_matrix(test_df['diagnosed_diabetes'], predictions)
    sns.heatmap(cm, annot=True)
    plt.title("Confusion Matrix XG Boost")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    print(cm)

    recall_xgboost = cm[1][1] / (cm[1][1] + cm[1][0])
    print(f"Recall for XG Boost: {recall_xgboost}")

################# Model Comparison #################
print(f"Decision Tree Accuracy: {decision_tree_accuracy}")
print(f"KNN Accuracy: {knn_accuracy}")
print(f"Random Forest Accuracy: {random_forest_accuracy}")
print(f"XG Boost Accuracy: {xgboost_accuracy}")
best_model = max(decision_tree_accuracy, knn_accuracy, random_forest_accuracy, xgboost_accuracy)

print(f"Decision Tree Recall: {recall_decision_tree}")
print(f"KNN Recall: {recall_knn}")
print(f"Random Forest Recall: {recall_random_forest}")
print(f"XG Boost Recall: {recall_xgboost}")

if best_model == decision_tree_accuracy:
    best_model_name = "Decision Tree"
elif best_model == knn_accuracy:
    best_model_name = "KNN"
elif best_model == random_forest_accuracy:
    best_model_name = "Random Forest"
elif best_model == xgboost_accuracy:
    best_model_name = "XG Boost"
print(f"\nBest Model: {best_model_name}")
print(f"{best_model_name} Accuracy: {best_model}")
with open("model_best.txt", "w") as f:
    f.write(best_model_name)