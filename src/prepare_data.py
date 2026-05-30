import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

def prepare_data(csv_path='data/raw/bank_clients.csv'):
    """Loading and preparing data from bank_clients.csv"""
    df = pd.read_csv(csv_path)
    
    print(f"The original size: {df.shape}")
    
    # 1. Deleting uninformative columns
    cols_to_drop = ['RowNumber', 'CustomerId', 'Surname']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # 2. Encoding categorical features
    # Gender: Male -> 0, Female -> 1
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    # Removing geography (has only 'Germany')
    if 'Geography' in df.columns and df['Geography'].nunique() == 1:
        df = df.drop(columns=['Geography'])
        print("The Geography feature has been removed (only one value)")
    
    # 3. Removing outliers by EstimatedSalary
    lower = df['EstimatedSalary'].quantile(0.01)
    upper = df['EstimatedSalary'].quantile(0.99)
    df = df[(df['EstimatedSalary'] >= lower) & (df['EstimatedSalary'] <= upper)]
    print(f"After removing the emissions: {df.shape}")
    
    # 4. Logarithmizing the skewed features
    df['Age_log'] = np.log1p(df['Age'])
    df['Balance_log'] = np.log1p(df['Balance'])
    df['Salary_log'] = np.log1p(df['EstimatedSalary'])
    
    # 5. Feature Engineering
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)
    df['TenureByAge'] = df['Tenure'] / (df['Age'] + 1)
    df['CreditScoreGivenAge'] = df['CreditScore'] / (df['Age'] + 1)
    
    # 6. Selecting the final signs
    feature_cols = [
        'CreditScore', 'Age_log', 'Tenure', 'Balance_log', 'NumOfProducts',
        'HasCrCard', 'IsActiveMember', 'Salary_log', 'Gender',
        'BalanceSalaryRatio', 'TenureByAge', 'CreditScoreGivenAge'
    ]
    
    # Checking which speakers are available
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols]
    y = df['Exited']
    
    print(f"\n{X.shape[0]} samples, {X.shape[1]} features")
    print(f"Features: {list(X.columns)}")
    print(f"Target: 0={sum(y==0)}, 1={sum(y==1)}")
    
    return X, y

if __name__ == "__main__":
    X, y = prepare_data()
    
    X.to_parquet('data/processed/X_features.parquet')
    y.to_parquet('data/processed/y_target.parquet')
    print("\nДанные сохранены в data/processed/")
