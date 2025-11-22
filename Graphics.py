import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class Graphics:
    def __init__(self, df):
        self.df = df
    
    def show_histograms(self, columns=None, cols_per_row=3):
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
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        plt.tight_layout()
        plt.show()
    
    def show_correlation_matrix(self, columns):
        corr_matrix = self.df[columns].corr()
        sns.heatmap(corr_matrix, annot=True)
        plt.show()
    def show_correlation_matrix_all(self):
        corr_matrix = self.df.corr()
        sns.heatmap(corr_matrix, annot=True)
        plt.show()
    
    def show_scatter_matrix(self, columns):
        sns.pairplot(self.df)
        plt.show()

    def show_boxplots(self, columns):
        self.df[columns].boxplot(figsize=(20,15))
        plt.show()
    
    def compare_hist(self, column1, column2):
        self.df[column1].hist(bins=50, figsize=(20,15))
        self.df[column2].hist(bins=50, figsize=(20,15))
        plt.show()
    
    def heatmap(self, columns):
        sns.heatmap(self.df[columns].corr(), annot=True)
        plt.show()