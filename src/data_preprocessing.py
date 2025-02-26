import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import os

def preprocess_data():
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    print("Loading data...")
    # Load data
    df = pd.read_csv('data/hr_data.csv')
    
    print("Preprocessing data...")
    # Convert target variable to binary
    df['Attrition_Binary'] = (df['Attrition'] == 'Yes').astype(int)
    
    # Drop unnecessary columns
    columns_to_drop = ['EmployeeCount', 'EmployeeNumber', 'StandardHours', 'Over18', 'Attrition']
    df = df.drop(columns=columns_to_drop)
    
    # Feature engineering
    print("Performing feature engineering...")
    
    # 1. Satisfaction composite score
    df['SatisfactionScore'] = (df['JobSatisfaction'] + 
                              df['EnvironmentSatisfaction'] + 
                              df['WorkLifeBalance'] + 
                              df['RelationshipSatisfaction']) / 4
    
    # 2. Years per company before joining
    df['YearsPerCompany'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)
    
    # 3. Salary to job level ratio (compensation fairness)
    df['SalaryToJobLevelRatio'] = df['MonthlyIncome'] / df['JobLevel']
    
    # 4. Career advancement ratio
    df['CareerAdvancementRatio'] = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    
    # 5. Overtime stress indicator
    df['OvertimeStress'] = (df['OverTime'] == 'Yes').astype(int) * (5 - df['WorkLifeBalance'])
    
    # Split features and target
    X = df.drop('Attrition_Binary', axis=1)
    y = df['Attrition_Binary']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Found {len(categorical_cols)} categorical features and {len(numerical_cols)} numerical features")
    
    # Create preprocessor
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # Split data
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit preprocessor on training data only
    print("Applying preprocessing transformations...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save preprocessor
    print("Saving preprocessor and processed data...")
    with open('models/preprocessor.pkl', 'wb') as f:
        pickle.dump(preprocessor, f)
    
    # Save feature names (useful for feature importance)
    categorical_feature_names = []
    if categorical_cols:
        encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        categorical_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
    
    feature_names = numerical_cols + categorical_feature_names
    
    with open('models/feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    
    # Save original column names for later reference
    with open('models/original_columns.pkl', 'wb') as f:
        pickle.dump(X.columns.tolist(), f)
    
    # Save processed datasets
    np.save('data/processed/X_train.npy', X_train_processed)
    np.save('data/processed/X_test.npy', X_test_processed)
    np.save('data/processed/y_train.npy', y_train.values)
    np.save('data/processed/y_test.npy', y_test.values)
    
    print(f"Data preprocessing completed. Processed data shape: {X_train_processed.shape}")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == "__main__":
    preprocess_data()