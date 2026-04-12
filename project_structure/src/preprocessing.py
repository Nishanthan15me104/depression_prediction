"""
Preprocessing module containing custom scikit-learn transformers.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DepressionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for cleaning text, constructing features, 
    and applying target encoding for the depression dataset.
    """
    def __init__(self):
        """Initializes empty dictionaries to store learned state during the fit process."""
        self.target_encodings = {}
        self.medians = {}
        
    def fit(self, X, y=None):
        """
        Learns parameters (medians, target encodings) from the training data 
        to prevent data leakage into the test set.
        """
        df = X.copy()
        if y is not None:
            df['target'] = y
            # Target encode Name and City
            self.target_encodings['Name'] = df.groupby('Name')['target'].mean()
            self.target_encodings['City'] = df.groupby('City')['target'].mean()
            
        # Learn medians for imputation
        self.medians['Name'] = self.target_encodings.get('Name', pd.Series()).median()
        self.medians['City'] = self.target_encodings.get('City', pd.Series()).median()
        self.medians['Financial Stress'] = df['Financial Stress'].mode()[0]
        
        # Learn CGPA mean for students
        self.medians['CGPA_Student'] = df[df['Working Professional or Student'] == 'Student']['CGPA'].mean()
        
        return self

    def transform(self, X):
        """
        Applies learned transformations and static cleaning rules to the dataset.
        """
        df = X.copy()
        
        # 1. Target Encoding Mapping
        if 'Name' in df.columns and 'Name' in self.target_encodings:
            df['Name'] = df['Name'].map(self.target_encodings['Name']).fillna(self.medians['Name'])
        if 'City' in df.columns and 'City' in self.target_encodings:
            df['City'] = df['City'].map(self.target_encodings['City']).fillna(self.medians['City'])

        # 2. Basic Imputations
        df['Financial Stress'] = df['Financial Stress'].fillna(self.medians['Financial Stress'])
        
        # Handle CGPA
        df['CGPA'] = df.apply(
            lambda row: np.nan if row['Working Professional or Student'] == 'Working Professional' else row['CGPA'], 
            axis=1
        )
        df['CGPA'] = df.apply(
            lambda row: self.medians['CGPA_Student'] if (row['Working Professional or Student'] == 'Student' and pd.isnull(row['CGPA'])) else row['CGPA'], 
            axis=1
        )
        
        # 3. Clean String Columns
        df['Cleaned_Sleep_Duration'] = df['Sleep Duration'].apply(self._clean_sleep_duration)
        df['Cleaned_Profession'] = df['Profession'].apply(self._group_profession)
        df['Cleaned_Degree'] = df['Degree'].apply(self._group_degree)
        
        # Dietary habits consolidation
        dietary_values = ['Healthy', 'Unhealthy', 'Moderate']
        df['Dietary Habits'] = df['Dietary Habits'].fillna('Moderate')
        df['Dietary Habits'] = df['Dietary Habits'].apply(lambda x: x if x in dietary_values else 'Moderate')

        # 4. Feature Construction (Ratios and Indices)
        df['Pressure'] = df.apply(
            lambda row: row['Academic Pressure'] if pd.isna(row['Work Pressure']) else row['Work Pressure'], axis=1
        )
        df['Satisfaction'] = df.apply(
            lambda row: row['Study Satisfaction'] if pd.isna(row['Job Satisfaction']) else row['Job Satisfaction'], axis=1
        )
        
        # Fill NaNs created by formulas with -1 to keep numerical integrity
        cols_to_fill = ['Academic Pressure', 'Study Satisfaction', 'Work Pressure', 'Job Satisfaction']
        for col in cols_to_fill:
            if col in df.columns:
                df[col] = df[col].fillna(-1)

        # 5. Drop original columns
        cols_to_drop = ['Name', 'City', 'CGPA', 'Sleep Duration', 'Degree', 'Profession']
        df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')
        
        return df

    # --- Static helper methods from notebook ---
    @staticmethod
    def _clean_sleep_duration(value):
        """Cleans and groups sleep duration text into standard bins."""
        if value in ['More than 8 hours']: return 'more than 8 hours'
        elif value in ['Less than 5 hours', '1-2 hours', '2-3 hours', '3-4 hours','4-5 hours']: return 'Less than 5 hours'
        elif value in ['5-6 hours', '4-6 hours','1-6 hours', '3-6 hours']: return '5-6 hours'
        elif value in ['7-8 hours']: return '7-8 hours'
        return 'Other'

    @staticmethod
    def _group_profession(profession):
        """Consolidates granular job titles into broader industry sectors."""
        if pd.isna(profession) or str(profession).lower() == 'unemployed': return 'Unemployed'
        prof = str(profession).lower()
        if prof in ['it', 'software engineer', 'data scientist', 'ux/ui designer', 'digital marketer']: return 'IT'
        if prof in ['teacher', 'educational consultant', 'researcher', 'academic']: return 'Education'
        if prof in ['financial analyst', 'accountant', 'investment banker']: return 'Finance'
        if prof in ['civil engineer', 'mechanical engineer', 'architect']: return 'Engineering'
        if prof in ['business analyst', 'marketing manager', 'entrepreneur', 'sales executive']: return 'Business'
        if prof in ['doctor', 'pharmacist', 'medical doctor']: return 'Healthcare'
        if prof in ['lawyer', 'judge']: return 'Law'
        if prof in ['chef', 'electrician', 'plumber']: return 'Skilled Trades'
        if prof in ['customer support', 'consultant']: return 'Other Services'
        return 'Other'

    @staticmethod
    def _group_degree(degree):
        """Standardizes degree types into major educational levels."""
        if pd.isna(degree) or isinstance(degree, (int, float)): return 'Other'
        deg = str(degree)
        if deg.startswith('B'): return 'Bachelor'
        if deg.startswith('Class'): return 'schooling'
        if deg.startswith('M'): return 'Master'
        if deg.startswith('L'): return 'Law'
        if deg.startswith('P'): return 'Diploma'
        return 'Other'