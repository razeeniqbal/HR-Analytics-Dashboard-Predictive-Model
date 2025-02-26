import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output directories
os.makedirs('output/eda', exist_ok=True)

def run_eda():
    # Load data
    print("Loading data...")
    df = pd.read_csv('data/hr_data.csv')
    
    # Basic information
    print("Dataset shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values:\n", df.isnull().sum())
    
    # Summary statistics
    print("\nSummary statistics:")
    print(df.describe())
    
    # Attrition distribution
    print("\nAttrition distribution:")
    attrition_counts = df['Attrition'].value_counts()
    print(attrition_counts)
    attrition_rate = attrition_counts['Yes'] / len(df) * 100
    print(f"Attrition rate: {attrition_rate:.2f}%")
    
    plt.figure(figsize=(8, 6))
    plt.pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', colors=['skyblue', 'salmon'])
    plt.title('Employee Attrition Distribution')
    plt.savefig('output/eda/attrition_distribution.png')
    plt.close()
    
    # Attrition by department
    plt.figure(figsize=(10, 6))
    dept_attrition = df.groupby('Department')['Attrition'].apply(lambda x: (x == 'Yes').mean() * 100)
    dept_attrition.plot(kind='bar', color='teal')
    plt.title('Attrition Rate by Department')
    plt.ylabel('Attrition Rate (%)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('output/eda/attrition_by_department.png')
    plt.close()
    
    # Age distribution by attrition
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x='Age', hue='Attrition', kde=True, bins=20)
    plt.title('Age Distribution by Attrition')
    plt.savefig('output/eda/age_distribution.png')
    plt.close()
    
    # Job satisfaction vs attrition
    plt.figure(figsize=(10, 6))
    job_sat_attrition = pd.crosstab(df['JobSatisfaction'], 
                                   df['Attrition'], 
                                   normalize='index') * 100
    job_sat_attrition['Yes'].plot(kind='bar', color='coral')
    plt.title('Attrition Rate by Job Satisfaction Level')
    plt.xlabel('Job Satisfaction Level')
    plt.ylabel('Attrition Rate (%)')
    plt.tight_layout()
    plt.savefig('output/eda/job_satisfaction_attrition.png')
    plt.close()
    
    # Monthly income vs attrition
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='Attrition', y='MonthlyIncome')
    plt.title('Monthly Income by Attrition')
    plt.savefig('output/eda/income_vs_attrition.png')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(16, 12))
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    correlation = numerical_df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    sns.heatmap(correlation, annot=False, mask=mask, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    plt.savefig('output/eda/correlation_heatmap.png')
    plt.close()
    
    # Overtime impact
    plt.figure(figsize=(10, 6))
    overtime_attrition = pd.crosstab(df['OverTime'], 
                                     df['Attrition'], 
                                     normalize='index') * 100
    overtime_attrition.plot(kind='bar', stacked=True)
    plt.title('Impact of Overtime on Attrition')
    plt.xlabel('Overtime')
    plt.ylabel('Percentage')
    plt.tight_layout()
    plt.savefig('output/eda/overtime_impact.png')
    plt.close()
    
    print("EDA completed. Visualizations saved to 'output/eda' directory.")

if __name__ == "__main__":
    run_eda()