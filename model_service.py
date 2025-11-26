import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from Class_implementations import DataProcessor

class DiabetesModelService:
    def __init__(self):
        self.model = None
        self.min_vals = None
        self.max_vals = None
        self.cat_mappings = {}
        # Important attributes based on code.py
        self.important_attributes = ["hba1c", "glucose_postprandial", "glucose_fasting", "family_history_diabetes", "age", "bmi", "systolic_bp", "smoking_status", "employment_status", "education_level", "income_level"]
        self.target_col = "diagnosed_diabetes"
        
    def load_and_train(self, csv_path="diabetes_dataset.csv", model_type="Auto"):
        """
        Loads and trains the model.
        Args:
            csv_path: Path to the CSV file containing the dataset.
            model_type: Type of model to use. Can be "Auto", "Random Forest", "Decision Tree", "KNN", "XG Boost".
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset not found at {csv_path}")
            
        print("Loading and processing data...")
        df = pd.read_csv(csv_path)
        
        # Initial cleaning
        df = df.dropna()
        if "diabetes_risk_score" in df.columns:
            df = df.drop("diabetes_risk_score", axis=1)
            
        # Remove outliers using DataProcessor
        # We need to identify numerical columns for outlier removal
        # In code.py, specific columns are checked
        cols_outlier_check = ["age", "alcohol_consumption_per_week", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day", "hypertension_history", "cardiovascular_history", "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides", "glucose_fasting", "glucose_postprandial", "insulin_level", "hba1c"]
        
        processor = DataProcessor(df)
        processor.remove_outliers(cols_outlier_check, threshold=2.0)
        df = processor.get_data()
        
        # Subsample data like in code.py (using 5% for speed, or more if needed)
        # code.py uses 0.05 for training
        processor = DataProcessor(df)
        df_subsampled = processor.subsample_data(fraction=0.1) # Using 10% for better service quality
        
        # Store min/max for normalization (using the subsampled data or full data? 
        # Ideally full data min/max is better, but let's use subsampled to be consistent with training data)
        # Actually, we should calculate min/max on the data we use for training.
        
        # Identify numerical columns for normalization
        num_cols = df_subsampled.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in num_cols:
            num_cols.remove(self.target_col)
            
        self.min_vals = df_subsampled[num_cols].min()
        self.max_vals = df_subsampled[num_cols].max()
        
        # Normalize
        # We do manual normalization here to ensure we use the same logic as we will in predict
        # Or use DataProcessor and trust it matches. 
        # DataProcessor does: (val - min) / (max - min)
        for col in num_cols:
            if self.max_vals[col] - self.min_vals[col] != 0:
                df_subsampled[col] = (df_subsampled[col] - self.min_vals[col]) / (self.max_vals[col] - self.min_vals[col])
            else:
                df_subsampled[col] = 0
                
        # Encoding
        # Drop gender = Other
        if 'gender' in df_subsampled.columns:
            df_subsampled = df_subsampled[df_subsampled['gender'] != 'Other']
            
        # Create mappings for categorical columns
        cat_cols = df_subsampled.select_dtypes(include=['object']).columns
        for col in cat_cols:
            # Create mapping: sorted unique values -> index
            unique_vals = sorted(df_subsampled[col].unique())
            mapping = {val: i for i, val in enumerate(unique_vals)}
            self.cat_mappings[col] = mapping
            # Apply mapping
            df_subsampled[col] = df_subsampled[col].map(mapping)
            
        # Prepare X and y
        # Ensure all important attributes are present
        missing_attrs = [attr for attr in self.important_attributes if attr not in df_subsampled.columns]
        if missing_attrs:
            raise ValueError(f"Missing attributes in dataset: {missing_attrs}")
            
        X = df_subsampled[self.important_attributes]
        y = df_subsampled[self.target_col]
        
        # Model Selection
        if model_type == "Auto":
            # Default to Random Forest as it performed best in analysis
            model_name = "Random Forest"
        else:
            model_name = model_type
            
        print(f"Training {model_name}...")
        if model_name == "Random Forest":
            self.model = RandomForestClassifier(n_estimators=100)
        elif model_name == "Decision Tree":
            self.model = DecisionTreeClassifier()
        elif model_name == "KNN":
            self.model = KNeighborsClassifier(n_neighbors=3)
        elif model_name == "XG Boost":
            self.model = XGBClassifier()
        else:
            raise ValueError(f"Unknown model type: {model_name}")
            
        self.model.fit(X, y)
        print("Model trained successfully.")

    def predict_patient(self, patient_data: dict):
        """
        Predicts diabetes for a single patient.
        Args:
            patient_data: dict containing patient attributes. 
                          Must contain all keys in self.important_attributes.
        Returns:
            int: 1 if patient is predicted to have diabetes, 0 otherwise.
        """
        if self.model is None:
            raise Exception("Model not trained. Call load_and_train() first.")
            
        # Create DataFrame from input
        input_df = pd.DataFrame([patient_data])
        
        # Normalize numerical columns using stored min/max
        for col in self.min_vals.index:
            if col in input_df.columns:
                min_v = self.min_vals[col]
                max_v = self.max_vals[col]
                if max_v - min_v != 0:
                    input_df[col] = (input_df[col] - min_v) / (max_v - min_v)
                else:
                    input_df[col] = 0
                    
        # Encode categorical columns using stored mappings
        for col, mapping in self.cat_mappings.items():
            if col in input_df.columns:
                input_df[col] = input_df[col].map(mapping)
                if input_df[col].isnull().any():
                    input_df[col] = input_df[col].fillna(-1) 
                    
        # Select important attributes
        try:
            X_input = input_df[self.important_attributes]
        except KeyError as e:
            raise ValueError(f"Input data missing required attributes: {e}")
            
        # Do prediction
        prediction = self.model.predict(X_input)
        return int(prediction[0])

if __name__ == "__main__":
    # Example usage for testing
    service = DiabetesModelService()
    service.load_and_train()
    
    # Test prediction using test patient
    # We need to provide raw values, as the service handles normalization/encoding
    test_patient = {
        "hba1c": 5.0,
        "glucose_postprandial": 100,
        "glucose_fasting": 90,
        "family_history_diabetes": 0, 
        "age": 30,
        "bmi": 22,
        "systolic_bp": 120,
        "smoking_status": "Never",
        "employment_status": "Employed",
        "education_level": "Graduate",
        "income_level": "Middle"
    }
    
    try:
        pred = service.predict_patient(test_patient)
        print(f"Prediction for test patient: {pred}")
    except Exception as e:
        print(f"Prediction failed (expected if test data schema doesn't match): {e}")