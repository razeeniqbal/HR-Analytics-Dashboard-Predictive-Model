import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go

def load_data_and_models():
    """Load the dataset and trained models"""
    # Load original data
    df = pd.read_csv('data/hr_data.csv')
    
    # Load the preprocessor
    with open('models/preprocessor.pkl', 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load model results
    with open('models/model_results.pkl', 'rb') as f:
        model_results = pickle.load(f)
    
    # Load best model (Random Forest or GradientBoosting usually)
    best_model_name = max(model_results.items(), key=lambda x: x[1]['f1'])[0]
    best_model = joblib.load(f'models/{best_model_name}_model.pkl')
    
    # Load feature names
    with open('models/feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # Load original column names
    with open('models/original_columns.pkl', 'rb') as f:
        original_columns = pickle.load(f)
    
    return df, preprocessor, best_model, model_results, feature_names, original_columns, best_model_name

def display_overview(df):
    st.title("HR Analytics Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_employees = len(df)
    attrition_count = df[df['Attrition'] == 'Yes'].shape[0]
    attrition_rate = attrition_count / total_employees * 100
    avg_satisfaction = df['JobSatisfaction'].mean()
    avg_tenure = df['YearsAtCompany'].mean()
    
    with col1:
        st.metric("Total Employees", f"{total_employees:,}")
    with col2:
        st.metric("Attrition Rate", f"{attrition_rate:.1f}%")
    with col3:
        st.metric("Avg. Job Satisfaction", f"{avg_satisfaction:.1f}/4")
    with col4:
        st.metric("Avg. Tenure", f"{avg_tenure:.1f} years")
    
    # Department breakdown
    st.subheader("Workforce by Department")
    dept_counts = df['Department'].value_counts()
    dept_fig = px.pie(
        values=dept_counts.values, 
        names=dept_counts.index,
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(dept_fig, use_container_width=True)
    
    # Two columns for charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Attrition by Department")
        dept_attrition = df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        dept_attrition.columns = ['Department', 'Attrition Rate (%)']
        
        fig = px.bar(
            dept_attrition,
            x='Department',
            y='Attrition Rate (%)',
            color='Department',
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Monthly Income by Job Role")
        role_income = df.groupby('JobRole')['MonthlyIncome'].mean().sort_values().reset_index()
        
        fig = px.bar(
            role_income,
            x='MonthlyIncome',
            y='JobRole',
            orientation='h',
            color='MonthlyIncome',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Satisfaction factors
    st.subheader("Job Satisfaction Factors")
    satisfaction_cols = ['JobSatisfaction', 'EnvironmentSatisfaction', 
                         'WorkLifeBalance', 'RelationshipSatisfaction']
    
    sat_data = df.groupby('Attrition')[satisfaction_cols].mean().reset_index()
    sat_data_melted = pd.melt(
        sat_data, 
        id_vars='Attrition',
        value_vars=satisfaction_cols,
        var_name='Satisfaction Type',
        value_name='Average Score'
    )
    
    fig = px.bar(
        sat_data_melted,
        x='Satisfaction Type',
        y='Average Score',
        color='Attrition',
        barmode='group',
        title='Satisfaction Factors by Attrition'
    )
    st.plotly_chart(fig, use_container_width=True)

def display_department_analysis(df):
    st.title("Department Analysis")
    
    # Department selector
    department = st.selectbox(
        "Select Department",
        df['Department'].unique()
    )
    
    # Filter data for the selected department
    dept_data = df[df['Department'] == department]
    
    # Key metrics comparison with company average
    st.subheader(f"Key Metrics: {department} vs. Company Average")
    
    # Create metrics dataframe for comparison
    metrics = pd.DataFrame({
        'Metric': ['Attrition Rate (%)', 'Avg Monthly Income', 'Avg Job Satisfaction', 'Avg Years at Company'],
        'Department': [
            dept_data['Attrition'].eq('Yes').mean() * 100,
            dept_data['MonthlyIncome'].mean(),
            dept_data['JobSatisfaction'].mean(),
            dept_data['YearsAtCompany'].mean()
        ],
        'Company Average': [
            df['Attrition'].eq('Yes').mean() * 100,
            df['MonthlyIncome'].mean(),
            df['JobSatisfaction'].mean(),
            df['YearsAtCompany'].mean()
        ]
    })
    
    # Calculate difference
    metrics['Difference'] = metrics['Department'] - metrics['Company Average']
    metrics['Percent Difference'] = (metrics['Difference'] / metrics['Company Average'] * 100)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta = f"{metrics.iloc[0]['Difference']:.1f}pp" # percentage points
        st.metric("Attrition Rate", f"{metrics.iloc[0]['Department']:.1f}%", delta=delta)
    
    with col2:
        delta = f"${metrics.iloc[1]['Difference']:.0f}"
        st.metric("Avg Monthly Income", f"${metrics.iloc[1]['Department']:.0f}", delta=delta)
    
    with col3:
        delta = f"{metrics.iloc[2]['Difference']:.2f}"
        st.metric("Avg Job Satisfaction", f"{metrics.iloc[2]['Department']:.2f}/4", delta=delta)
    
    with col4:
        delta = f"{metrics.iloc[3]['Difference']:.1f} years"
        st.metric("Avg Tenure", f"{metrics.iloc[3]['Department']:.1f} years", delta=delta)
    
    # Department visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Job Roles Distribution")
        role_counts = dept_data['JobRole'].value_counts()
        
        fig = px.pie(
            values=role_counts.values,
            names=role_counts.index,
            title=f"Job Roles in {department}"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Attrition by Job Role")
        role_attrition = dept_data.groupby('JobRole')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        role_attrition.columns = ['JobRole', 'Attrition Rate (%)']
        
        fig = px.bar(
            role_attrition,
            x='JobRole',
            y='Attrition Rate (%)',
            title=f"Attrition Rate by Job Role in {department}",
            color='Attrition Rate (%)',
            color_continuous_scale='Reds'
        )
        fig.update_layout(xaxis={'categoryorder':'total descending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Overtime analysis
    st.subheader("Overtime Impact Analysis")
    
    # Calculate overtime data
    overtime_data = dept_data.groupby('OverTime')['Attrition'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).reset_index()
    overtime_data.columns = ['OverTime', 'Attrition Rate (%)']
    
    # Additional stats
    overtime_counts = dept_data['OverTime'].value_counts(normalize=True) * 100
    overtime_pct = overtime_counts.get('Yes', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Employees Working Overtime", f"{overtime_pct:.1f}%")
        
        fig = px.pie(
            values=dept_data['OverTime'].value_counts().values,
            names=dept_data['OverTime'].value_counts().index,
            title="Overtime Distribution",
            color_discrete_sequence=['#91cf60', '#fc8d59']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            overtime_data,
            x='OverTime',
            y='Attrition Rate (%)',
            title="Attrition Rate by Overtime Status",
            color='Attrition Rate (%)',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)

def display_attrition_prediction(df, preprocessor, model, original_columns):
    st.title("Employee Attrition Prediction")
    
    st.write("""
    This tool helps predict the likelihood of an employee leaving the organization. 
    Adjust the parameters below to see how different factors affect attrition risk.
    """)
    
    # Create form for input
    st.subheader("Employee Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=60, value=35)
        gender = st.selectbox("Gender", df['Gender'].unique())
        marital_status = st.selectbox("Marital Status", df['MaritalStatus'].unique())
        department = st.selectbox("Department", df['Department'].unique())
        job_role = st.selectbox("Job Role", df[df['Department'] == department]['JobRole'].unique())
    
    with col2:
        job_level = st.slider("Job Level", min_value=1, max_value=5, value=2)
        monthly_income = st.slider("Monthly Income ($)", min_value=1000, max_value=20000, value=5000, step=500)
        job_satisfaction = st.slider("Job Satisfaction", min_value=1, max_value=4, value=3)
        work_life_balance = st.slider("Work Life Balance", min_value=1, max_value=4, value=3)
        overtime = st.selectbox("Works Overtime?", ['Yes', 'No'])
    
    with col3:
        years_at_company = st.slider("Years at Company", min_value=0, max_value=40, value=5)
        years_in_role = st.slider("Years in Current Role", min_value=0, max_value=20, value=3)
        distance_from_home = st.slider("Distance From Home (miles)", min_value=1, max_value=30, value=10)
        total_working_years = st.slider("Total Working Years", min_value=0, max_value=40, value=10)
        years_since_promotion = st.slider("Years Since Last Promotion", min_value=0, max_value=15, value=2)
    
    # Create input data dictionary
    input_data = {
        'Age': age,
        'Gender': gender,
        'MaritalStatus': marital_status,
        'Department': department,
        'JobRole': job_role,
        'JobLevel': job_level,
        'MonthlyIncome': monthly_income,
        'JobSatisfaction': job_satisfaction,
        'WorkLifeBalance': work_life_balance,
        'OverTime': overtime,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_role,
        'DistanceFromHome': distance_from_home,
        'TotalWorkingYears': total_working_years,
        'YearsSinceLastPromotion': years_since_promotion,
        
        # Set default values for remaining features
        'BusinessTravel': 'Travel_Rarely',
        'Education': 3,
        'EducationField': 'Life Sciences',
        'EnvironmentSatisfaction': 3,
        'NumCompaniesWorked': 2,
        'PercentSalaryHike': 15,
        'PerformanceRating': 3,
        'RelationshipSatisfaction': 3,
        'StockOptionLevel': 1,
        'TrainingTimesLastYear': 2,
        'YearsWithCurrManager': years_in_role
    }
    
    # Create DataFrame from input data
    input_df = pd.DataFrame([input_data])
    
    # Add engineered features (same as in preprocessing script)
    input_df['SatisfactionScore'] = (input_df['JobSatisfaction'] + 
                                   input_df['EnvironmentSatisfaction'] + 
                                   input_df['WorkLifeBalance'] + 
                                   input_df['RelationshipSatisfaction']) / 4
    
    input_df['YearsPerCompany'] = input_df['TotalWorkingYears'] / (input_df['NumCompaniesWorked'] + 1)
    input_df['SalaryToJobLevelRatio'] = input_df['MonthlyIncome'] / input_df['JobLevel']
    input_df['CareerAdvancementRatio'] = input_df['YearsSinceLastPromotion'] / (input_df['YearsAtCompany'] + 1)
    input_df['OvertimeStress'] = (input_df['OverTime'] == 'Yes').astype(int) * (5 - input_df['WorkLifeBalance'])
    
    # Make prediction when button is clicked
    if st.button("Predict Attrition Risk"):
        # Ensure input DataFrame has the same columns as original data
        for col in original_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # default value
        
        # Reorder columns to match original data
        input_df = input_df[original_columns]
        
        # Transform input data
        input_processed = preprocessor.transform(input_df)
        
        # Get prediction and probability
        attrition_pred = model.predict(input_processed)[0]
        attrition_prob = model.predict_proba(input_processed)[0, 1]
        
        # Display prediction
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gauge chart for attrition probability
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=attrition_prob * 100,
                title={'text': "Attrition Risk"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "gray"},
                    'steps': [
                        {'range': [0, 30], 'color': "green"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            risk_level = "Low" if attrition_prob < 0.3 else "Medium" if attrition_prob < 0.7 else "High"
            st.write(f"**Risk Level:** {risk_level}")
        
        with col2:
            # Key risk factors
            st.subheader("Key Risk Factors")
            
            risk_factors = {
                "Low Job Satisfaction": 4 - job_satisfaction,
                "Poor Work-Life Balance": 4 - work_life_balance,
                "Overtime": 1 if overtime == "Yes" else 0,
                "Distance from Home": distance_from_home / 30,
                "Low Relative Salary": 1 - (monthly_income / (10000 * job_level)),
                "Limited Career Growth": years_since_promotion / 5
            }
            
            # Filter and sort risk factors
            significant_factors = {k: v for k, v in risk_factors.items() if v > 0.4}
            sorted_factors = dict(sorted(significant_factors.items(), key=lambda x: x[1], reverse=True))
            
            if sorted_factors:
                for factor, value in list(sorted_factors.items())[:3]:
                    st.write(f"• {factor}")
            else:
                st.write("No significant risk factors identified.")
        
        # Recommendations
        st.subheader("Retention Recommendations")
        if attrition_prob > 0.7:
            st.write("⚠️ **High attrition risk detected!** Immediate intervention recommended:")
            st.write("1. Schedule a one-on-one career development discussion")
            st.write("2. Review compensation package")
            st.write("3. Implement workload management plan")
        elif attrition_prob > 0.4:
            st.write("⚠️ **Moderate attrition risk detected.** Consider the following actions:")
            st.write("1. Check in on job satisfaction and career goals")
            st.write("2. Provide learning and development opportunities")
            st.write("3. Review work-life balance concerns")
        else:
            st.write("✅ **Low attrition risk.** Continue with regular engagement:")
            st.write("1. Maintain regular feedback and recognition")
            st.write("2. Include in succession planning")
            st.write("3. Offer mentoring opportunities")

def display_model_insights(model, feature_names, model_results, best_model_name):
    st.title("Model Insights")
    
    # Model performance
    st.subheader("Model Performance Comparison")
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame({
        'Model': [],
        'Metric': [],
        'Value': []
    })
    
    for model_name, results in model_results.items():
        for metric, value in results.items():
            if metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                metrics_df = pd.concat([
                    metrics_df,
                    pd.DataFrame({
                        'Model': [model_name],
                        'Metric': [metric.capitalize()],
                        'Value': [value]
                    })
                ], ignore_index=True)
    
    # Plot model comparison
    fig = px.bar(
        metrics_df,
        x='Model',
        y='Value',
        color='Metric',
        barmode='group',
        title='Model Performance Metrics'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"**Best Model:** {best_model_name} (based on F1 Score)")
    
    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        
        # Get feature importances
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Create dataframe for visualization
        top_n = min(20, len(feature_names))
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices[:top_n]],
            'Importance': importances[indices[:top_n]]
        })
        
        # Plot feature importance
        fig = px.bar(
            importance_df,
            x='Importance',
            y='Feature',
            orientation='h',
            title=f'Top {top_n} Feature Importances',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top 5 features explanation
        st.subheader("Key Drivers of Attrition")
        top_features = importance_df.head(5)['Feature'].tolist()
        
        explanations = {
            'OverTime_Yes': "Employees who work overtime are more likely to leave the company, potentially due to burnout and work-life balance issues.",
            'MonthlyIncome': "Compensation is a critical factor in employee retention. Lower salaries correlate with higher attrition rates.",
            'JobLevel': "Job level impacts attrition, with employees at lower job levels showing higher turnover rates.",
            'Age': "Age is a significant predictor of attrition, with younger employees generally having higher turnover rates.",
            'TotalWorkingYears': "Less experienced employees tend to change jobs more frequently.",
            'YearsAtCompany': "Employees with fewer years at the company are at higher risk of leaving.",
            'DistanceFromHome': "Longer commutes correlate with higher attrition rates.",
            'JobSatisfaction': "Lower job satisfaction strongly predicts employee turnover.",
            'WorkLifeBalance': "Poor work-life balance is a key driver of employee attrition.",
            'SalaryToJobLevelRatio': "Employees who feel underpaid relative to their job level are more likely to leave.",
            'JobRole_Sales Representative': "Sales Representatives show higher turnover compared to other roles.",
            'CareerAdvancementRatio': "Limited career advancement opportunities drive attrition.",
            'Department_Sales': "The Sales department typically experiences higher turnover rates.",
            'YearsSinceLastPromotion': "Employees who haven't been promoted recently are at higher risk of leaving.",
            'OvertimeStress': "The combination of overtime work and low work-life balance significantly increases attrition risk."
        }
        
        for feature in top_features:
            for key in explanations:
                if key in feature:
                    st.write(f"**{feature}**: {explanations[key]}")
                    break
            else:
                st.write(f"**{feature}**: This feature has a significant impact on predicting employee attrition.")

def main():
    # Page config
    st.set_page_config(
        page_title="HR Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    # Load data and models
    df, preprocessor, model, model_results, feature_names, original_columns, best_model_name = load_data_and_models()
    
    # Sidebar navigation
    st.sidebar.title("HR Analytics Dashboard")
    page = st.sidebar.radio(
        "Navigate to", 
        ["Overview", "Department Analysis", "Attrition Prediction", "Model Insights"]
    )
    
    if page == "Overview":
        display_overview(df)
    elif page == "Department Analysis":
        display_department_analysis(df)
    elif page == "Attrition Prediction":
        display_attrition_prediction(df, preprocessor, model, original_columns)
    elif page == "Model Insights":
        display_model_insights(model, feature_names, model_results, best_model_name)

if __name__ == "__main__":
    main()