# 👥 People Analytics Dashboard

## Overview
Interactive analytics dashboard providing insights into employee attrition, engagement, and performance drivers.

## Business Problem
Employee turnover costs companies 50-200% of annual salary. This dashboard helps HR teams:
- Identify at-risk employees before they leave
- Understand key drivers of attrition
- Make data-driven retention decisions

## Key Findings
1. **Top Attrition Drivers**: Low monthly income, younger age, and short tenure
2. **High-Risk Segment**: Employees with 0-2 years tenure have **28.9% attrition rate** (3.5x higher than long-tenured employees)
3. **Retention Lever**: Job satisfaction is statistically proven to differ between employees who stay vs. leave (p < 0.001)

## Features
- 📊 Interactive workforce overview
- 🎯 Attrition analysis by department, role, and tenure
- 💼 Performance & engagement insights with statistical validation
- 🔮 ML-powered attrition predictions with feature importance

## Tech Stack
- **Data Processing**: Python, Pandas, NumPy
- **Modeling**: Scikit-learn, Random Forest
- **Statistical Analysis**: SciPy
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Deployment**: Streamlit Cloud (coming soon)

## Model Performance
- **Accuracy**: 75%
- **ROC-AUC**: 0.670
- **Top Predictor**: Monthly Income (25.3% importance)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/people-analytics-dashboard.git

# Navigate to project directory
cd people-analytics-dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app/streamlit_app.py
```

## Project Structure
```
people-analytics-dashboard/
├── data/
│   ├── WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── cleaned_data.csv
├── notebooks/
│   └── Data Analysis & Insights.ipynb
├── models/
│   └── attrition_model.pkl
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## Future Enhancements
- Real-time data integration
- Sentiment analysis from employee surveys
- Predictive prescriptive recommendations
- A/B testing for retention interventions

## Author
**Tasnim Tabassum**  
Data Scientist | Machine Learning Engineer  
[LinkedIn](https://www.linkedin.com/in/towmony) | [Portfolio](https://tasnim-data-scientist.github.io/)

## Dataset
IBM HR Analytics Employee Attrition Dataset (Kaggle)