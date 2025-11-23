import pandas as pd
import random

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
        return self.df.iloc[selected_indexes].reset_index(drop=True)

    
