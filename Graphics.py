import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Graphics:
    def __init__(self, df):
        self.df = df
    
    def show_histograms(self, columns):
        self.df[columns].hist(bins=50, figsize=(20,15))
        plt.show()
    
    def show_correlation_matrix(self, columns):
        corr_matrix = self.df[columns].corr()
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