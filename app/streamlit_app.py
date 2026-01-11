import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import sys
import os

# Add parent directory to path to import from notebooks
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page config
st.set_page_config(
    page_title="People Analytics Dashboard",
    page_icon="👥",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    # Get the project root directory
    import os
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(root_dir, 'data', 'cleaned_data.csv')
    df = pd.read_csv(data_path)
    return df

@st.cache_resource
def load_model():
    import os
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(root_dir, 'models', 'attrition_model.pkl')
    return joblib.load(model_path)

# Load data and model
try:
    df = load_data()
    model = load_model()
    st.success("✅ Data and model loaded successfully!")
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Main title
st.title("👥 People Analytics Dashboard")
st.markdown("""
    **Interactive insights into workforce attrition, performance, and engagement**
    
    This dashboard analyzes employee data to identify attrition patterns and predict at-risk employees.
    Use the filters in the sidebar to explore different departments.
""")

st.markdown("---")

# Display basic info
st.write(f"Dataset contains {len(df)} employees")

# Sidebar filters
st.sidebar.header("📊 Filters")
dept_filter = st.sidebar.multiselect(
    "Select Department(s)",
    options=df['Department'].unique(),
    default=df['Department'].unique()
)

# Filter data based on selection
filtered_df = df[df['Department'].isin(dept_filter)]

st.markdown("---")

# KPIs row
st.subheader("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_employees = len(filtered_df)
    st.metric("Total Employees", total_employees)

with col2:
    attrition_rate = (filtered_df['Attrition'] == 'Yes').mean() * 100
    st.metric("Attrition Rate", f"{attrition_rate:.1f}%")

with col3:
    avg_tenure = filtered_df['YearsAtCompany'].mean()
    st.metric("Avg Tenure", f"{avg_tenure:.1f} years")

with col4:
    avg_satisfaction = filtered_df['JobSatisfaction'].mean()
    st.metric("Avg Job Satisfaction", f"{avg_satisfaction:.1f}/4")
    

st.markdown("---")

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", 
    "🎯 Attrition Analysis", 
    "💼 Performance & Engagement",
    "🔮 Predictions"
])

with tab1:
    st.subheader("Workforce Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Department distribution
        dept_counts = filtered_df['Department'].value_counts()
        fig = px.pie(
            values=dept_counts.values,
            names=dept_counts.index,
            title="Employees by Department"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Age distribution
        fig = px.histogram(
            filtered_df,
            x='Age',
            nbins=20,
            title="Age Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🎯 Attrition Deep Dive")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Attrition by Department
        attrition_by_dept = filtered_df.groupby('Department')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        attrition_by_dept.columns = ['Department', 'Attrition_Rate']
        
        fig = px.bar(
            attrition_by_dept,
            x='Department',
            y='Attrition_Rate',
            title='Attrition Rate by Department',
            labels={'Attrition_Rate': 'Attrition Rate (%)'},
            color='Attrition_Rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Attrition by Job Role
        attrition_by_role = filtered_df.groupby('JobRole')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        attrition_by_role.columns = ['JobRole', 'Attrition_Rate']
        attrition_by_role = attrition_by_role.sort_values('Attrition_Rate', ascending=False)
        
        fig = px.bar(
            attrition_by_role,
            x='Attrition_Rate',
            y='JobRole',
            orientation='h',
            title='Attrition Rate by Job Role',
            labels={'Attrition_Rate': 'Attrition Rate (%)'},
            color='Attrition_Rate',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Attrition by Tenure
    st.markdown("### Attrition by Tenure")
    tenure_attrition = filtered_df.groupby('TenureGroup')['Attrition'].apply(
        lambda x: (x == 'Yes').mean() * 100
    ).reset_index()
    tenure_attrition.columns = ['TenureGroup', 'Attrition_Rate']
    
    fig = px.bar(
        tenure_attrition,
        x='TenureGroup',
        y='Attrition_Rate',
        title='Attrition Rate by Tenure Group',
        labels={'Attrition_Rate': 'Attrition Rate (%)', 'TenureGroup': 'Tenure Group'},
        color='Attrition_Rate',
        color_continuous_scale='Oranges'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Key insight
    st.success("💡 **Key Finding**: Employees with 0-2 years tenure have the highest attrition rate (28.9%)")

with tab3:
    st.subheader("💼 Performance & Engagement Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Satisfaction by Attrition
        fig = px.box(
            filtered_df,
            x='Attrition',
            y='OverallSatisfaction',
            title='Overall Satisfaction by Attrition Status',
            labels={'OverallSatisfaction': 'Overall Satisfaction Score', 'Attrition': 'Employee Status'},
            color='Attrition',
            color_discrete_map={'Yes': 'red', 'No': 'green'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Work-Life Balance
        wlb_attrition = filtered_df.groupby('WorkLifeBalance')['Attrition'].apply(
            lambda x: (x == 'Yes').mean() * 100
        ).reset_index()
        wlb_attrition.columns = ['WorkLifeBalance', 'Attrition_Rate']
        
        fig = px.line(
            wlb_attrition,
            x='WorkLifeBalance',
            y='Attrition_Rate',
            markers=True,
            title='Attrition Rate by Work-Life Balance',
            labels={'WorkLifeBalance': 'Work-Life Balance Score', 'Attrition_Rate': 'Attrition Rate (%)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistical insight
    st.markdown("### 📊 Statistical Analysis")
    
    from scipy import stats
    stayed = filtered_df[filtered_df['Attrition'] == 'No']['JobSatisfaction']
    left = filtered_df[filtered_df['Attrition'] == 'Yes']['JobSatisfaction']
    
    if len(stayed) > 0 and len(left) > 0:
        t_stat, p_value = stats.ttest_ind(stayed, left)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Satisfaction (Stayed)", f"{stayed.mean():.2f}")
        with col2:
            st.metric("Avg Satisfaction (Left)", f"{left.mean():.2f}")
        with col3:
            if p_value < 0.05:
                st.metric("Statistical Significance", "✅ YES", delta=f"p={p_value:.6f}")
            else:
                st.metric("Statistical Significance", "❌ NO", delta=f"p={p_value:.6f}")
        
        if p_value < 0.05:
            st.success("💡 **Key Finding**: Job satisfaction is significantly lower for employees who left (statistically proven)")
    else:
        st.warning("Not enough data for statistical analysis with current filters")

with tab4:
    st.subheader("🔮 Attrition Risk Predictions")
    
    # Model performance
    st.markdown("### 📊 Model Performance")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", "75%")
    with col2:
        st.metric("ROC-AUC Score", "0.670")
    with col3:
        st.metric("Recall (Left)", "36%")
    
    st.info("💡 The model correctly identifies 36% of employees who actually left, with 75% overall accuracy")
    
    # Feature importance
    st.markdown("### 🎯 Top Attrition Predictors")
    
    feature_importance = pd.DataFrame({
        'Feature': ['MonthlyIncome', 'Age', 'YearsAtCompany', 'DistanceFromHome', 
                   'OverallSatisfaction', 'NumCompaniesWorked', 'WorkLifeBalance'],
        'Importance': [0.253, 0.177, 0.171, 0.104, 0.088, 0.082, 0.048]
    })
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Feature Importance for Attrition Prediction',
        labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("💡 **Key Finding**: Monthly Income is the #1 predictor of attrition (25.3% importance)")
    
    # High-risk employees
    st.markdown("### ⚠️ High-Risk Employee Segments")
    
    # Calculate risk scores
    high_risk = filtered_df[
        ((filtered_df['YearsAtCompany'] <= 2) & 
         (filtered_df['JobSatisfaction'] <= 2)) |
        (filtered_df['OverallSatisfaction'] <= 2)
    ]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("High-Risk Employees", len(high_risk))
        st.metric("% of Workforce", f"{len(high_risk)/len(filtered_df)*100:.1f}%")
    
    with col2:
        if len(high_risk) > 0:
            high_risk_attrition = (high_risk['Attrition'] == 'Yes').mean() * 100
            st.metric("Actual Attrition in High-Risk Group", f"{high_risk_attrition:.1f}%")
            st.metric("vs Overall Attrition", f"{attrition_rate:.1f}%", 
                     delta=f"{high_risk_attrition - attrition_rate:.1f}%", 
                     delta_color="inverse")
    
# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>👩‍💻 Built by Tasnim Tabassum | Data Scientist | 
        <a href='https://www.linkedin.com/in/towmony' target='_blank'>LinkedIn</a> | 
        <a href='https://tasnim-data-scientist.github.io/' target='_blank'>Portfolio</a></p>
        <p style='font-size: 12px;'>People Analytics Dashboard • Python • Streamlit • Machine Learning</p>
    </div>
""", unsafe_allow_html=True)