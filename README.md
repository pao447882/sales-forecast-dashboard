# Sales Forecast Dashboard

## Challenge
Small and Medium Enterprises (SMEs) often lack tools for data-driven decisions, relying on intuition due to limited resources and data analysis skills.

## Solution
This project developed an **all-in-one sales forecasting and decision-support dashboard**.  
We collected **3 years of sales and external factors data**, then built three machine learning models:

- **Linear Regression** → quantitative sales forecasts  
- **Prophet** → time-series forecasting  
- **Logistic Regression** → classify "high-sales days"

The best results were displayed on an **interactive dashboard (Flask + HTML)**, providing easy-to-understand and actionable insights.

## Impact
The tool empowers SME owners with accurate forecasts, improving inventory and cost management.

## Key Features
- Time series forecasting (Prophet, Linear Regression)
- Classification for sales events (Logistic Regression)
- Data preprocessing & feature engineering
- Model evaluation (MAE, RMSE, F1-score)
- Interactive Flask dashboard
- Integration with Google Sheets API for data updates

## Tech Stack
- **Python** (Pandas, NumPy, Scikit-learn, Prophet)  
- **Flask + HTML/CSS** (interactive dashboard)  
- **Jupyter Notebooks** (experiments & EDA)  
- **Google Sheets API** (data integration)  
- **Matplotlib/Seaborn** (visualizations)  

## Repository Structure  
sales-forecast-dashboard/  
│── data/ # Raw & processed data (sample only, anonymized)  
│── notebooks/ # Jupyter Notebooks (EDA, experiments, models)  
│ ├── 0.1_Preprocess.ipynb  
│ ├── LinearRegression_Phase6_Recursive.ipynb  
│ ├── LinearRegression_Phase6_Direct.ipynb  
│ ├── Prophet_Experiment6.ipynb  
│ ├── Prophet_Experiment7_lag7.ipynb  
│ ├── LogisticRegression_Phase8_Recursive.ipynb  
│ ├── LogisticRegression_Phase8_Direct.ipynb  
│ ├── LogisticRegression_Phase8_lag17.ipynb  
│── dashboard/ # Flask app files  
│ ├── app.py  
│ ├── templates/  
│ │ └── dashboard.html  
│ ├── static/  
│ ├── style.css  
│ └── scripts.js  
│── models/ # Trained models (.pkl files, optional, small only)  
│── requirements.txt # Project dependencies  
│── README.md # Project overview  
│── .gitignore # Ignore sensitive/large files  

## Results
- Linear Regression (Recursive) → MAE = 1312.15
- Prophet → MAE = 2634.97
- Logistic Regression (Direct) → F1 = 1.00
- All results integrated into a user-friendly dashboard to support SME decision-making.

## Notes
-- Raw business data is excluded for privacy.
-- Only sample or anonymized datasets are provided.



