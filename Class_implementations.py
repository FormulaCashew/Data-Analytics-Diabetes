import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix

class Graphics:
    def __init__(self, df):
        self.df = df
    
    def show_histograms(self, columns=None, cols_per_row=3):
        '''
        Function to show histograms for the given columns
        Accepts both numerical and categorical columns
        
        Args: 
            columns: list of columns to show histograms for
            cols_per_row: number of columns per row
        
        Returns:
        None
        '''
        sns.set(style="whitegrid")
        if columns is None:
            columns = self.df.columns

        if len(columns) == 0:
            print("No columns provided")
            return

        # Check for existing columns in the dataset
        columns = [col for col in columns if col in self.df.columns]

        num_cols = len(columns)
        num_rows = int(np.ceil(num_cols / cols_per_row))
        fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row*4, num_rows*4))
        axes = axes.flatten()

        for i, column in enumerate(columns):
            ax = axes[i]
            if self.df[column].dtype == 'object':
                sns.countplot(self.df[column], ax=ax)
            else:
                sns.histplot(self.df[column], bins=50, ax=ax)
            ax.set_title(column)
            ax.set_xlabel(column)
            ax.set_ylabel("Frequency")
            ax.tick_params(axis='x', rotation=45) # Rotate x-axis labels
        
        # Hide unused axes
        for i in range(num_cols, num_rows * cols_per_row):
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()
    def show_correlation_matrix(self, columns):
        '''
        Function to show correlation matrix for the given columns
        Values given must be numerical
        
        Args: 
            columns: list of columns to show correlation matrix for
        
        Returns:
        None
        '''
        if not all(col in self.df.select_dtypes(include=['float64', 'int64', 'int8']).columns for col in columns):
            raise ValueError("All columns must be numerical")
        if len(columns) > 10:
            print("Too many columns, may overlap")
        elif len(columns) == 0:
            print("No columns provided")
            return
        corr_matrix = self.df[columns].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.show()
    def show_correlation_matrix_all(self):
        if not all(col in self.df.select_dtypes(include=['float64', 'int64', 'int8']).columns for col in self.df.columns):
            raise ValueError("All columns must be numerical")
        if len(self.df.columns) > 10:
            # Show only colors
            print("Too many columns to show correlation matrix, showing only colors")
            self.heatmap(self.df.columns)
        else:
            # Show correlation matrix for all columns
            self.show_correlation_matrix(self.df.columns)
    def show_scatter_matrix(self, columns):
        '''
        Function to show scatter matrix for the given columns
        Values given must be numerical
        
        Args: 
            columns: list of columns to show scatter matrix for
        
        Returns:
            None
        '''
        if not all(col in self.df.select_dtypes(include=['float64', 'int64', 'int8']).columns for col in columns):
            raise ValueError("All columns must be numerical")
        if len(columns) > 5:
            # Show only colors
            print("Too many columns to show scatter matrix")
            return
        else:
            sns.pairplot(self.df[columns])
            print("Showing scatter matrix for columns:", columns)
            plt.show()
    def show_boxplots(self, columns, cols_per_row=3):
        '''
        Function to show boxplots for the given columns
        Values given must be numerical
        
        Args: 
            columns: list of columns to show boxplots
            cols_per_row: number of columns per row
        
        Returns:
            None
        '''
        if not all(col in self.df.select_dtypes(include=['float64', 'int64', 'int8']).columns for col in columns):
            raise ValueError("All columns must be numerical")
        
        columns = [col for col in columns if col in self.df.columns]
        num_cols = len(columns)
        num_rows = int(np.ceil(num_cols / cols_per_row))
        fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row*4, num_rows*4))
        axes = axes.flatten()
        for i, col in enumerate(columns):
            ax = axes[i]
            sns.boxplot(x=col, data=self.df, ax=ax)
            ax.set_xlabel(col)
            ax.set_ylabel("Frequency")
        plt.tight_layout()
        plt.show()
    def compare_hist(self, column1, column2):
        '''
        Function to compare histograms for the given columns, frequency of each column is shown
        Values given must be numerical
        
        Args: 
            column1: first column to compare
            column2: second column to compare
        
        Returns:
            None
        '''
        plt.figure(figsize=(20,15))
        self.df[column1].hist(bins=50)
        self.df[column2].hist(bins=50)
        plt.legend([column1, column2])
        plt.show()
    def show_hist_axis(self, x_axis, y_axis):
        '''
        Function to show histogram using x and y axis from columns of the dataframe
        Values given must be numerical
        
        Args: 
            x_axis: column to show histogram for
            y_axis: column to show histogram for
        
        Returns:
            None
        '''
        self.df[x_axis].hist(bins=50)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.show()
    def heatmap(self, columns, cols_per_row=3):
        '''
        Function to show heatmap for the given columns
        Values given must be numerical
        
        Args: 
            columns: list of columns to show heatmap for
        
        Returns:
        None
        '''
        if not all(col in self.df.select_dtypes(include=['float64', 'int64', 'int8']).columns for col in columns):
            raise ValueError("All columns must be numerical")
        
        # Calculate figure size based on number of columns
        n = len(columns)
        figsize = (max(10, n * 0.6), max(8, n * 0.5))
        
        plt.figure(figsize=figsize)
        sns.heatmap(self.df[columns].corr(), cmap='coolwarm', xticklabels=True, yticklabels=True)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

class DataProcessor:
    def __init__(self, df):
        self.df = df.copy()

    def subsample_data(self, samples=None, fraction=None, random_state=None):
        """
        Subsample the dataset
        
        Args: 
            samples: Number of samples to return
            fraction: Fraction of axis items to return
            random_state: Seed for random number generator
        
        Returns:
        Subsampled dataframe
        """
        # Set random seed
        if random_state is not None:
            random.seed(random_state)
        
        total_rows = len(self.df)
        indexes = list(range(total_rows))
        
        # Check if samples or fraction is provided
        if samples is not None:
            k = samples
        elif fraction is not None:
            k = int(total_rows * fraction)
        else:
            raise ValueError("Please provide either samples or fraction")
        
        # Check if sample size is larger than number of rows
        if k > total_rows:
            raise ValueError("Sample size cannot be larger than the number of rows")
        
        # Random selection of indexes
        selected_indexes = random.sample(indexes, k)
        
        # Using iloc to select rows based on the manually selected indexes
        print("Subsampled down to {} rows from {} rows".format(k, total_rows))
        return self.df.iloc[selected_indexes].reset_index(drop=True).copy()

    def normalize_data(self, columns):
        """
        Normalizes the data in the given columns values 0 to 1
        
        Args: 
            columns: list of columns to normalize
        
        Returns:
            Dataframe with normalized columns
        """
        for col in columns:
            self.df[col] = (self.df[col] - self.df[col].min()) / (self.df[col].max() - self.df[col].min())  # Normalize values to 0-1
        return self.df.copy()

    def get_data(self):
        """
        Returns the dataframe
        
        Returns:
            Dataframe
        """
        return self.df.copy()
    
    def plot_kmeans_clustering(self, columns, n_clusters=3, x_col=None, y_col=None):
        """
        Performs KMeans clustering and displays a scatter plot encoded by cluster color
        
        Args: 
            columns: list of columns to use for clustering
            n_clusters: number of clusters
            x_col: column for x-axis (optional, defaults to first column in columns)
            y_col: column for y-axis (optional, defaults to second column in columns)
        
        Returns:
            None
        """
        if len(columns) < 2:
             raise ValueError("Need at least 2 columns for clustering and plotting")

        # Ensure data is numeric
        data = self.df[columns].select_dtypes(include=[np.number])
        if data.shape[1] != len(columns):
             raise ValueError("All columns must be numerical for KMeans")

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(data)
        
        if x_col is None:
            x_col = columns[0]
        if y_col is None:
            y_col = columns[1]
            
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.df[x_col], y=self.df[y_col], hue=labels, palette='viridis')
        plt.title(f'KMeans Clustering (k={n_clusters})')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend(title='Cluster')
        plt.show() 

    def train_test_split(self, test_size=0.2, random_state=None):
        """
        Splits the dataset into training and testing sets
        
        Args: 
            test_size: Proportion of the dataset to include in the test split (0.0 to 1.0)
            random_state: Seed for random number generator
        
        Returns:
            tuple: (train_df, test_df)
        """
        if random_state is not None:
            random.seed(random_state)
            
        total_rows = len(self.df)
        indexes = list(range(total_rows))
        
        # Shuffle indexes
        random.shuffle(indexes)
        
        # Calculate split index
        split_index = int(total_rows * (1 - test_size))
        
        # Select indexes for training and testing
        train_indexes = indexes[:split_index]
        test_indexes = indexes[split_index:]
        
        # Assign rows to training and testing dataframes
        train_df = self.df.iloc[train_indexes].reset_index(drop=True).copy()
        test_df = self.df.iloc[test_indexes].reset_index(drop=True).copy()
        
        return train_df, test_df 

class KNN:
    """
    Class KNN_Model with methods to implement KNN_Model

    """

    def __init__(self, k: int = 3):
        """
        Constructor of KNN_Model
        Args:
            k (int): number of neighbors to use, defaults to 3, it is recommended to use an odd number
        """
        self._outputs_train = None
        self._inputs_train = None
        self._k = k

    def set_k(self, k: int):
        """ Simple setter for k neighbors"""
        self._k = k

    def store(self, inputs_train, outputs_train):
        """
        Stores training data
        Args:
            inputs_train (pandas.DataFrame): training data inputs
            outputs_train (pandas.DataFrame): training data outputs
        """
        self._inputs_train = np.array(inputs_train)
        self._outputs_train = np.array(outputs_train)
        print("Training data stored")

    def predict(self, inputs):
        """
        Predicts an output given various inputs data
        Args:
            inputs (pandas.DataFrame): input data
        Returns:
            pandas.DataFrame: predicted outputs
        """
        inputs_arr = np.array(inputs)
        predictions = [self.predict_single(input_row) for input_row in inputs_arr] # List with various outputs
        return predictions

    def predict_single(self, input_test):
        """
        Function to predict the output given single input data row
        Args:
            input_test (pandas.DataFrame): input data, needs to have the Attributes of the training data
        Returns:
            pandas.DataFrame: predicted outputs
        """
        distances = []
        for i, input_row in enumerate(self._inputs_train):
            dist = self.euclidean_distance(input_row, input_test)   # measure distance between given inputs and local data
            distances.append((dist, self._outputs_train[i])) # save distance and output data

        sorted_distances = sorted(distances, key=lambda x: x[0])
        nearest_neighbors = sorted_distances[:self._k] # get only the distances up to k
        nearest_neighbor_labels = [neighbor[1] for neighbor in nearest_neighbors]
        prediction = Counter(nearest_neighbor_labels).most_common(1)[0] # Check for most common output in the neighbors
        return prediction[0]
    
    def plot_confusion_matrix(self, inputs, outputs):
        """
        Plots a confusion matrix for the given inputs and outputs
        Args:
            inputs (pandas.DataFrame): input data
            outputs (pandas.DataFrame): output data
        """
        predictions = self.predict(inputs)
        cm = confusion_matrix(outputs, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        print(cm)


    @staticmethod
    def euclidean_distance(p1, p2):
        return np.sqrt(np.sum((p1 - p2)**2))