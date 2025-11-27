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
        
    def load_and_train(self, csv_path="diabetes_dataset.csv"):
        """
        Loads, processes, and trains multiple models to select the best one dynamically.
        Args:
            csv_path: Path to the CSV file containing the dataset.
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
        cols_outlier_check = ["age", "alcohol_consumption_per_week", "physical_activity_minutes_per_week", "diet_score", "sleep_hours_per_day", "screen_time_hours_per_day", "hypertension_history", "cardiovascular_history", "bmi", "waist_to_hip_ratio", "systolic_bp", "diastolic_bp", "heart_rate", "cholesterol_total", "hdl_cholesterol", "ldl_cholesterol", "triglycerides", "glucose_fasting", "glucose_postprandial", "insulin_level", "hba1c"]
        
        processor = DataProcessor(df)
        processor.remove_outliers(cols_outlier_check, threshold=2.0)
        df = processor.get_data()
        
        # Subsample data (using 5% for speed as requested)
        processor = DataProcessor(df)
        df_subsampled = processor.subsample_data(fraction=0.05)
        
        # Identify numerical columns for normalization
        num_cols = df_subsampled.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_col in num_cols:
            num_cols.remove(self.target_col)
            
        self.min_vals = df_subsampled[num_cols].min()
        self.max_vals = df_subsampled[num_cols].max()
        
        # Normalize
        for col in num_cols:
            if self.max_vals[col] - self.min_vals[col] != 0:
                df_subsampled[col] = (df_subsampled[col] - self.min_vals[col]) / (self.max_vals[col] - self.min_vals[col])
            else:
                df_subsampled[col] = 0
                
        # Encoding
        if 'gender' in df_subsampled.columns:
            df_subsampled = df_subsampled[df_subsampled['gender'] != 'Other']
            
        # Create mappings for categorical columns
        cat_cols = df_subsampled.select_dtypes(include=['object']).columns
        for col in cat_cols:
            unique_vals = sorted(df_subsampled[col].unique())
            mapping = {val: i for i, val in enumerate(unique_vals)}
            self.cat_mappings[col] = mapping
            df_subsampled[col] = df_subsampled[col].map(mapping)
            
        # Prepare X and y
        missing_attrs = [attr for attr in self.important_attributes if attr not in df_subsampled.columns]
        if missing_attrs:
            raise ValueError(f"Missing attributes in dataset: {missing_attrs}")
            
        # Split data for model evaluation
        processor = DataProcessor(df_subsampled)
        train_df, test_df = processor.train_test_split(test_size=0.2)
        
        X_train = train_df[self.important_attributes]
        y_train = train_df[self.target_col]
        X_test = test_df[self.important_attributes]
        y_test = test_df[self.target_col]
        
        # Define models to evaluate
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "Decision Tree": DecisionTreeClassifier(),
            "KNN": KNeighborsClassifier(n_neighbors=3),
            "XG Boost": XGBClassifier()
        }
        
        best_accuracy = 0
        best_model_name = ""
        
        print("\nEvaluating models...")
        for name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = sum(predictions == y_test) / len(y_test)
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                self.model = model
                
        print(f"\nBest Model Selected: {best_model_name} with Accuracy: {best_accuracy:.4f}")
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
    test_patient = {
        "hba1c": 5.0, # A low value, indicating good blood sugar control
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
        print(f"Prediction failed: {e}")