import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans

class Graphics:
    def __init__(self, df):
        self.df = df
    
    def show_histograms(self, columns=None, cols_per_row=3):
        '''
        Function to show histograms for the given columns
        Accepts both numerical and categorical columns
        
        Parameters:
        columns: list of columns to show histograms for
        cols_per_row: number of columns per row
        
        Returns:
        None
        '''
        sns.set(style="whitegrid")
        if columns is None:
            columns = self.df.columns
        
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
    def show_correlation_matrix(self, columns, cols_per_row=3):
        '''
        Function to show correlation matrix for the given columns
        Values given must be numerical
        
        Parameters:
        columns: list of columns to show correlation matrix for
        
        Returns:
        None
        '''
        if not all(col in self.df.select_dtypes(include=['float64', 'int64', 'int8']).columns for col in columns):
            raise ValueError("All columns must be numerical")
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
    def show_scatter_matrix(self, columns, cols_per_row=3):
        '''
        Function to show scatter matrix for the given columns
        Values given must be numerical
        
        Parameters:
        columns: list of columns to show scatter matrix for
        
        Returns:
        None
        '''
        if not all(col in self.df.select_dtypes(include=['float64', 'int64', 'int8']).columns for col in self.df.columns):
            raise ValueError("All columns must be numerical")
        sns.pairplot(self.df[columns])
        plt.show()
    def show_boxplots(self, columns, cols_per_row=3):
        '''
        Function to show boxplots for the given columns
        Values given must be numerical
        
        Parameters:
        columns: list of columns to show boxplots for
        
        Returns:
        None
        '''
        if not all(col in self.df.select_dtypes(include=['float64', 'int64', 'int8']).columns for col in columns):
            raise ValueError("All columns must be numerical")
        self.df[columns].boxplot(figsize=(20,15))
        plt.show()
    def compare_hist(self, column1, column2):
        plt.figure(figsize=(20,15))
        plt.subplot(1,2,1)
        self.df[column1].hist(bins=50)
        plt.subplot(1,2,2)
        self.df[column2].hist(bins=50)
        plt.show()
    def heatmap(self, columns, cols_per_row=3):
        '''
        Function to show heatmap for the given columns
        Values given must be numerical
        
        Parameters:
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
        self.df = df

    def subsample_data(self, samples=None, fraction=None, random_state=None):
        """
        Subsample the dataset
        
        Parameters:
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
        return self.df.iloc[selected_indexes].reset_index(drop=True)

    def plot_kmeans_clustering(self, columns, n_clusters=3, x_col=None, y_col=None):
        """
        Performs KMeans clustering and displays a scatter plot encoded by cluster color
        
        Parameters:
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
        
        Parameters:
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
        train_df = self.df.iloc[train_indexes].reset_index(drop=True)
        test_df = self.df.iloc[test_indexes].reset_index(drop=True)
        
        return train_df, test_df 
