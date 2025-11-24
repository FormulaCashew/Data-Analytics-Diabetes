# Data Analysis for Diabetes Risk

This project is a data analysis project for diabetes risk, it uses the diabetes_dataset.csv file to analyze the data and provide insights about the risk of diabetes. The dataset is downloaded from kaggle directly from the code using the kagglehub library.

## Data

The data used for this analysis is not included in the repository to reduce the size. It is automatically downloaded when running the code. The data was generated synthetically but based on real patient data; it contains information about 100k people and their risk of diabetes.

## Project Structure

- `code.py`: The main script that performs data loading, cleaning, exploratory data analysis (EDA), normalization, encoding, and modeling.
- `Class_implementations.py`: Contains custom classes used in the main script.
- `requirements.txt`: List of Python dependencies.

## Classes (`Class_implementations.py`)

The code uses the following custom classes:

- **Graphics**: A class that provides methods for visualizing data, including histograms, correlation matrices, scatter matrices, and boxplots.
- **DataProcessor**: A class that provides various methods for processing data, such as subsampling, normalization, and train-test splitting.
- **KNN**: A custom implementation of the K-Nearest Neighbors algorithm.

## Workflow

The `code.py` script follows these steps:

1. **Data Loading**: Downloads the dataset from Kaggle if not present.
2. **Initial Data Cleaning**: Checks for missing values and drops unnecessary columns.
3. **EDA Visualization**: Visualizes data distributions and correlations.
4. **Normalization**: Normalizes numerical data to a 0-1 range.
5. **Encoding**: Encodes categorical variables into numeric codes.
6. **KMeans Clustering**: Performs KMeans clustering on selected features.
7. **Modeling**:
   - **Decision Tree**: Uses `sklearn.tree.DecisionTreeClassifier` for classification.
   - **KNN**: Compares `sklearn.neighbors.KNeighborsClassifier` with the custom `KNN` class.

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Usage

To run the analysis:

```bash
python code.py
```
