import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

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
        if not all(col in self.df.select_dtypes(include=['float64', 'int64', 'int8']).columns for col in columns):
            raise ValueError("All columns must be numerical")
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.show()
    
    def show_scatter_matrix(self, columns, cols_per_row=3):
        '''
        Function to show scatter matrix for the given columns
        Values given must be numerical
        
        Parameters:
        columns: list of columns to show scatter matrix for
        
        Returns:
        None
        '''
        if not all(self.df[col].dtype == 'float64' or self.df[col].dtype == 'int64' for col in columns):
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
        if not all(self.df[col].dtype == 'float64' or self.df[col].dtype == 'int64' for col in columns):
            raise ValueError("All columns must be numerical")
        self.df[columns].boxplot(figsize=(20,15))
        plt.show()
    
    def compare_hist(self, column1, column2):
        self.df[column1].hist(bins=50, figsize=(20,15))
        self.df[column2].hist(bins=50, figsize=(20,15))
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
        if not all(self.df[col].dtype == 'float64' or self.df[col].dtype == 'int64' for col in columns):
            raise ValueError("All columns must be numerical")
        sns.heatmap(self.df[columns].corr(), annot=True, cmap='coolwarm')
        plt.show()