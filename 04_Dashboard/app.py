# 1. IMPORT---------------------------------------------------------------------------
import datetime
import pandas as pd
import joblib
from prophet import Prophet
from flask import Flask, jsonify
from flask_cors import CORS
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from flask import Flask, render_template
from dashboard_export import generate_dashboard_html
from pytrends.request import TrendReq
import os
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from datetime import datetime
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from markupsafe import Markup
#--------------------------------------------------------------------------------------------------

# 2. FLASK APP INITIALIZATION--------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)
#-----------------------------------------------------------------------------------------------------

# 3. CONFIGURATIONS-----------------------------------------------------------------------------------------
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("/var/www/html/credentials.json", scope)
client = gspread.authorize(credentials)

# Load saved Prophet model using joblib
model_path = "/var/www/html/models/prophet_model.pkl"
prophet_model = joblib.load(model_path)
#-----------------------------------------------------------------------------------------------------------

# 4. DEF FUNCTION-------------------------------------------------------------------------------------------
# (4) Function to fetch data from Google Sheets
def get_sales_data():
    spreadsheet = client.open("nd_sales")
    sheet = spreadsheet.worksheet("dailysales")
    data = sheet.get_all_records()
    df = pd.DataFrame(data)
    return df
    
# (4) Prophet Data Preprocessing 
def preprocess_data_prophet(df):
    # Google Sheets CSV Export URL
    sheet_url = ""
    csv_url = sheet_url.replace("/edit?usp=sharing", "/export?format=csv")

    # Load data from Google Sheets CSV
    df = pd.read_csv(csv_url)

    # Drop rows with missing values
    df_cleaned = df.dropna()
    df = df_cleaned.copy()

    df['new_datetime'] = pd.to_datetime(df['date'])

    # Drop unwanted columns
    df = df.drop(columns=['local_event', 'other_event'])

    # Remove commas from numeric string columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.replace(',', '', regex=False)

    # Convert columns to appropriate data types
    df['day'] = df['day'].astype('object')
    df['date'] = pd.to_datetime(df['date'])
    df['cost_change'] = df['cost_change'].astype('float64')
    df['cost_material'] = df['cost_material'].astype('float64')
    df['cost_banking'] = df['cost_banking'].astype('float64')
    df['cost_sum'] = df['cost_sum'].astype('float64')
    df['sales_fs'] = df['sales_fs'].astype('float64')
    df['sales_qr'] = df['sales_qr'].astype('float64')
    df['sales_gb'] = df['sales_gb'].astype('float64')
    df['sales_sp'] = df['sales_sp'].astype('float64')
    df['sales_lm'] = df['sales_lm'].astype('float64')
    df['sales_rb'] = df['sales_rb'].astype('float64')
    df['sales_sum'] = df['sales_sum'].astype('float64')
    df['profit'] = pd.to_numeric(df['profit'], errors='coerce').astype('float64')
    df['promotion_apply'] = df['promotion_apply'].astype('object')
    df['ismarketday'] = df['ismarketday'].astype('int64')
    df['isschoolday'] = df['isschoolday'].astype('int64')
    df['holiday'] = df['holiday'].astype('int64')
    df['isbuddaday'] = df['isbuddaday'].astype('int64')
    df['nd_lottery'] = df['nd_lottery'].astype('int64')
    df['applylkcolor'] = df['applylkcolor'].astype('object')
    df['PP'] = df['PP'].astype('float64')
    df['T'] = df['T'].astype('float64')
    df['H'] = df['H'].astype('float64')
    df['V'] = df['V'].astype('float64')
    df['pm2.5'] = df['pm2.5'].astype('float64')
    df['pm10'] = df['pm10'].astype('float64')
    df['o3'] = df['o3'].astype('float64')
    df['no2'] = df['no2'].astype('float64')
    
    global df_to_calculate
    df_to_calculate = df.copy()
    
    # Merge cost_banking into cost_material and drop cost_banking
    df['cost_material'] = df['cost_material'] + df['cost_banking']
    df = df.drop(columns=['cost_banking'])

    # Merge sales_qr into sales_fs and drop sales_qr
    df['sales_qr'] = df['sales_qr'].fillna(0)
    df['sales_fs'] = df['sales_fs'].fillna(0)
    df['sales_fs'] = df['sales_qr'] + df['sales_fs']
    df = df.drop(columns=['sales_qr'])

    # Map Thai day abbreviations to English
    day_mapping = {
        'อา.': 'Sun',
        'จ.': 'Mon',
        'อ.': 'Tue',
        'พ.': 'Wed',
        'พฤ.': 'Thu',
        'ศ.': 'Fri',
        'ส.': 'Sat'
    }
    df['day'] = df['day'].replace(day_mapping)

    df.loc[df['day'] == 'Fri', 'sales_sum'] = df.loc[df['day'] == 'Fri', 'sales_sum'] + 7000

    df_prophet = df[['new_datetime', 'sales_sum', 'sales_fs', 'sales_gb', 'ismarketday', 'isschoolday', 'holiday', 'V']].copy()

    # Prepare expected column names
    df_prophet.columns = ['ds', 'y', 'add1', 'add2', 'add3', 'add4', 'add5', 'add6']
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    return df_prophet

# (4) Prophet Regression Sales Forecasting Function
def forecast_sales(df_prophet):
    import pandas as pd
    import joblib
    from datetime import datetime, timedelta

    # Load trained Prophet model
    model_path = "/var/www/html/models/prophet_model.pkl"  # Adjust path if needed
    prophet_model = joblib.load(model_path)

    # Ensure ds column is in datetime format
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    # Define the last available date in the dataset
    last_date = df_prophet['ds'].max()

    # Generate next 30 future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})

    # Get past 3 months of data
    start_train_date = last_date - timedelta(days=90)
    recent_data = df_prophet[df_prophet['ds'] >= start_train_date].copy()

    # Ensure 'day_of_week' column exists (0=Monday, 6=Sunday)
    recent_data['day_of_week'] = recent_data['ds'].dt.dayofweek
    future_df['day_of_week'] = future_df['ds'].dt.dayofweek  # Add same feature to future data

    # Assign average regressor values based on the same weekday in the past 3 months
    regressor_cols = ['add1', 'add2', 'add3', 'add4', 'add5', 'add6']

    for col in regressor_cols:
        for i, row in future_df.iterrows():
            # Get past 3 months' data for the same day of the week
            past_values = recent_data[recent_data['day_of_week'] == row['day_of_week']][col]

            if not past_values.empty:
                future_df.at[i, col] = past_values.mean()  # Assign the average value
            else:
                future_df.at[i, col] = recent_data[col].mean()  # Default to total mean if missing

    # Drop helper column
    future_df = future_df.drop(columns=['day_of_week'])

    # Print future data before prediction
    print("Future Data for Prediction:")
    print(future_df)

    # Make predictions using the loaded Prophet model
    forecast = prophet_model.predict(future_df)

    # Prevent negative sales predictions
    forecast["yhat"] = forecast["yhat"].apply(lambda x: max(x, 0))

    return forecast[['ds', 'yhat']].rename(columns={"ds": "date", "yhat": "predicted_sales"})

# (4) Linear Regression Data Preprocessing 
def preprocess_data_linear(df):
    # Drop rows with missing values
    df_cleaned = df.dropna()
    df = df_cleaned.copy()

    # Convert date columns
    df['new_datetime'] = pd.to_datetime(df['date'])

    # Drop unwanted columns
    df = df.drop(columns=['local_event', 'other_event'])

    # Remove commas from numeric strings
    for col in df.columns:
     if df[col].dtype == 'object':
          df[col] = df[col].astype(str).str.replace(',', '', regex=False)

    # Convert columns to appropriate data types
    df['day'] = df['day'].astype('object')
    df['date'] = pd.to_datetime(df['date'])
    df['cost_change'] = pd.to_numeric(df['cost_change'], errors='coerce').astype('float64')
    df['cost_material'] = pd.to_numeric(df['cost_material'], errors='coerce').astype('float64')
    df['cost_banking'] = pd.to_numeric(df['cost_banking'], errors='coerce').astype('float64')
    df['cost_sum'] = pd.to_numeric(df['cost_sum'], errors='coerce').astype('float64')
    df['sales_fs'] = pd.to_numeric(df['sales_fs'], errors='coerce').astype('float64')
    df['sales_qr'] = pd.to_numeric(df['sales_qr'], errors='coerce').astype('float64')
    df['sales_gb'] = pd.to_numeric(df['sales_gb'], errors='coerce').astype('float64')
    df['sales_sp'] = pd.to_numeric(df['sales_sp'], errors='coerce').astype('float64')
    df['sales_lm'] = pd.to_numeric(df['sales_lm'], errors='coerce').astype('float64')
    df['sales_rb'] = pd.to_numeric(df['sales_rb'], errors='coerce').astype('float64')
    df['sales_sum'] = pd.to_numeric(df['sales_sum'], errors='coerce').astype('float64')
    df['profit'] = pd.to_numeric(df['profit'], errors='coerce').astype('float64')
    df['promotion_apply'] = df['promotion_apply'].astype('object')
    df['ismarketday'] = pd.to_numeric(df['ismarketday'], errors='coerce').fillna(0).astype('int64')
    df['isschoolday'] = pd.to_numeric(df['isschoolday'], errors='coerce').fillna(0).astype('int64')
    df['holiday'] = pd.to_numeric(df['holiday'], errors='coerce').fillna(0).astype('int64')
    df['isbuddaday'] = pd.to_numeric(df['isbuddaday'], errors='coerce').fillna(0).astype('int64')
    df['nd_lottery'] = pd.to_numeric(df['nd_lottery'], errors='coerce').fillna(0).astype('int64')
    df['applylkcolor'] = df['applylkcolor'].astype('object')
    df['PP'] = pd.to_numeric(df['PP'], errors='coerce').astype('float64')
    df['T'] = pd.to_numeric(df['T'], errors='coerce').astype('float64')
    df['H'] = pd.to_numeric(df['H'], errors='coerce').astype('float64')
    df['V'] = pd.to_numeric(df['V'], errors='coerce').astype('float64')
    df['pm2.5'] = pd.to_numeric(df['pm2.5'], errors='coerce').astype('float64')
    df['pm10'] = pd.to_numeric(df['pm10'], errors='coerce').astype('float64')
    df['o3'] = pd.to_numeric(df['o3'], errors='coerce').astype('float64')
    df['no2'] = pd.to_numeric(df['no2'], errors='coerce').astype('float64')

    global df_to_calculate
    df_to_calculate = df.copy()

    # Merge banking cost into material cost
    df['cost_material'] = df['cost_material'] + df['cost_banking']
    df = df.drop(columns=['cost_banking'])

    # Merge QR sales into FS sales
    df['sales_qr'] = df['sales_qr'].fillna(0)
    df['sales_fs'] = df['sales_fs'].fillna(0)
    df['sales_fs'] = df['sales_qr'] + df['sales_fs']
    df = df.drop(columns=['sales_qr'])

    # Map Thai day abbreviations to English
    day_mapping = {
            'อา.': 'Sun',
            'จ.': 'Mon',
            'อ.': 'Tue',
            'พ.': 'Wed',
            'พฤ.': 'Thu',
            'ศ.': 'Fri',
            'ส.': 'Sat'
        }
    df['day'] = df['day'].replace(day_mapping)

    # Adjust Friday sales by adding 7000
    df.loc[df['day'] == 'Fri', 'sales_sum'] = df.loc[df['day'] == 'Fri', 'sales_sum'] + 7000
    
    df_lr = df.copy()
    df_lr = df_lr.dropna()
    df_lr_plot = df_lr.copy()
    print("df_lr_plot.shape")
    print(df_lr_plot.shape)
    print(df_lr.tail(10))

    # Return cleaned dataframe
    return df_lr.copy(), df_lr_plot.copy()
#--------------------------------------------------------------------------

# (4) Linear Regression Sales Forecasting Function
def forecast_sales_linear(df_lr):
    import numpy as np
    # Assuming df_lr is already defined as in your provided code.
    from datetime import timedelta
  
    # เก็บวันสุดท้ายของข้อมูลก่อนกรอง
    last_date_raw = pd.to_datetime(df_lr['date'], errors='coerce').max()
    last_date = last_date_raw


    # ค่อยกรองย้อนหลัง 90 วันภายหลัง
    ninety_days_ago = last_date - timedelta(days=90)
    df_lr = df_lr[df_lr['date'] >= ninety_days_ago]
    
    from sklearn.preprocessing import MinMaxScaler
    columns_to_normalize = ['sales_sum', 'sales_fs', 'sales_gb', 'V']
    scaler = MinMaxScaler()
    df_lr[columns_to_normalize] = scaler.fit_transform(df_lr[columns_to_normalize])

    # Ensure 'date' is datetime
    df_lr['date'] = pd.to_datetime(df_lr['date'], errors='coerce')

    # เพิ่ม day_of_week สำหรับกรอง
    df_lr['day_of_week'] = df_lr['date'].dt.dayofweek  # 0=Monday, 6=Sunday

    # Set lag
    lag = 7
    x_ = df_lr['sales_sum'].iloc[-lag:].values

    # Prepare คอลัมน์ที่ใช้
    regressor_cols = ['sales_fs', 'sales_gb', 'V']
    categorical_cols = ['ismarketday', 'isschoolday', 'holiday']

    # Load model ที่ฝึกไว้
    model_path = "/var/www/html/models/Linear_Regression_Model.pkl"
    model = joblib.load(model_path)

    df_lr['date'] = pd.to_datetime(df_lr['date'])
    df_lr['day'] = df_lr['day'].astype(str)

    # Set lag
    lag = 7
    x_ = df_lr['sales_sum'].iloc[-lag:].values

    # Prepare คอลัมน์ที่ใช้
    regressor_cols = ['sales_fs', 'sales_gb', 'V']

    # Prepare for prediction
    predictions = []
    future_dates = []
    future_days = []

    # ทำนาย 7 วันล่วงหน้า
    for i in range(1, 8):
        target_date = last_date + timedelta(days=i)
        target_dayname = target_date.strftime('%a')  # เช่น Sat, Sun

        # หาข้อมูลย้อนหลัง 3 เดือน (90 วัน)
        past_3_months = df_lr[df_lr['date'] >= (last_date - timedelta(days=90))]

        # คำนวณค่าเฉลี่ยของ regressors เฉพาะวันที่ตรงกัน
        regressor_avg = past_3_months[past_3_months['day'] == target_dayname][regressor_cols].mean()

        fs = regressor_avg['sales_fs']
        gb = regressor_avg['sales_gb']
        V = regressor_avg['V']

        # หาค่าของ categorical features จากข้อมูลล่าสุดที่ตรงกับวันในอดีต
        past_same_day = df_lr[df_lr['day'] == target_dayname].iloc[-1]
        ismarketday = past_same_day['ismarketday']
        isschoolday = past_same_day['isschoolday']
        holiday = past_same_day['holiday']

        # รวมเป็น feature vector (ไม่ต้อง scale แล้ว)
        x_input = x_.tolist() + [fs, gb, ismarketday, isschoolday, holiday, V]

        # Predict (ค่าดิบ)
        pred = model.predict([x_input])[0]
        predictions.append(pred)
        future_dates.append(target_date)
        future_days.append(target_dayname)

        # Update lag
        x_ = np.roll(x_, -1)
        x_[-1] = pred

    # ใช้ min-max ที่ scaler เก็บไว้จาก sales_sum
    sales_sum_min = scaler.data_min_[0]  # index 0 คือ sales_sum
    sales_sum_max = scaler.data_max_[0]

    # Denormalize แล้วจัดการค่าติดลบ
    denormalized_sales = []
    for pred in predictions:
        val = pred * (sales_sum_max - sales_sum_min) + sales_sum_min
        val = max(val, 0)  # ถ้าติดลบ ให้เป็น 0
        denormalized_sales.append(val)
    
    # สร้าง DataFrame ของผลลัพธ์
    df_forecast_linear = pd.DataFrame({
        'date': future_dates,
        'day': future_days,
        'predicted_sales_normalized': predictions,
        'predicted_sales': denormalized_sales  # ค่าจริงหลัง denormalize
    })
    
    return df_forecast_linear
#--------------------------------------------------------------------------

# (4) Logistic Regression Data Preprocessing 
def preprocess_data_logistic(df):
    # Drop rows with missing values
    df_cleaned = df.dropna()
    df = df_cleaned.copy()

    # Convert date columns
    df['new_datetime'] = pd.to_datetime(df['date'])

    # Drop unwanted columns
    df = df.drop(columns=['local_event', 'other_event'])

    # Remove commas from numeric strings
    for col in df.columns:
     if df[col].dtype == 'object':
          df[col] = df[col].astype(str).str.replace(',', '', regex=False)

    # Convert columns to appropriate data types
    df['day'] = df['day'].astype('object')
    df['date'] = pd.to_datetime(df['date'])
    df['cost_change'] = pd.to_numeric(df['cost_change'], errors='coerce').astype('float64')
    df['cost_material'] = pd.to_numeric(df['cost_material'], errors='coerce').astype('float64')
    df['cost_banking'] = pd.to_numeric(df['cost_banking'], errors='coerce').astype('float64')
    df['cost_sum'] = pd.to_numeric(df['cost_sum'], errors='coerce').astype('float64')
    df['sales_fs'] = pd.to_numeric(df['sales_fs'], errors='coerce').astype('float64')
    df['sales_qr'] = pd.to_numeric(df['sales_qr'], errors='coerce').astype('float64')
    df['sales_gb'] = pd.to_numeric(df['sales_gb'], errors='coerce').astype('float64')
    df['sales_sp'] = pd.to_numeric(df['sales_sp'], errors='coerce').astype('float64')
    df['sales_lm'] = pd.to_numeric(df['sales_lm'], errors='coerce').astype('float64')
    df['sales_rb'] = pd.to_numeric(df['sales_rb'], errors='coerce').astype('float64')
    df['sales_sum'] = pd.to_numeric(df['sales_sum'], errors='coerce').astype('float64')
    df['profit'] = pd.to_numeric(df['profit'], errors='coerce').astype('float64')
    df['promotion_apply'] = df['promotion_apply'].astype('object')
    df['ismarketday'] = pd.to_numeric(df['ismarketday'], errors='coerce').fillna(0).astype('int64')
    df['isschoolday'] = pd.to_numeric(df['isschoolday'], errors='coerce').fillna(0).astype('int64')
    df['holiday'] = pd.to_numeric(df['holiday'], errors='coerce').fillna(0).astype('int64')
    df['isbuddaday'] = pd.to_numeric(df['isbuddaday'], errors='coerce').fillna(0).astype('int64')
    df['nd_lottery'] = pd.to_numeric(df['nd_lottery'], errors='coerce').fillna(0).astype('int64')
    df['applylkcolor'] = df['applylkcolor'].astype('object')
    df['PP'] = pd.to_numeric(df['PP'], errors='coerce').astype('float64')
    df['T'] = pd.to_numeric(df['T'], errors='coerce').astype('float64')
    df['H'] = pd.to_numeric(df['H'], errors='coerce').astype('float64')
    df['V'] = pd.to_numeric(df['V'], errors='coerce').astype('float64')
    df['pm2.5'] = pd.to_numeric(df['pm2.5'], errors='coerce').astype('float64')
    df['pm10'] = pd.to_numeric(df['pm10'], errors='coerce').astype('float64')
    df['o3'] = pd.to_numeric(df['o3'], errors='coerce').astype('float64')
    df['no2'] = pd.to_numeric(df['no2'], errors='coerce').astype('float64')

    global df_to_calculate
    df_to_calculate = df.copy()

    # Merge banking cost into material cost
    df['cost_material'] = df['cost_material'] + df['cost_banking']
    df = df.drop(columns=['cost_banking'])

    # Merge QR sales into FS sales
    df['sales_qr'] = df['sales_qr'].fillna(0)
    df['sales_fs'] = df['sales_fs'].fillna(0)
    df['sales_fs'] = df['sales_qr'] + df['sales_fs']
    df = df.drop(columns=['sales_qr'])

    # Map Thai day abbreviations to English
    day_mapping = {
            'อา.': 'Sun',
            'จ.': 'Mon',
            'อ.': 'Tue',
            'พ.': 'Wed',
            'พฤ.': 'Thu',
            'ศ.': 'Fri',
            'ส.': 'Sat'
        }
    df['day'] = df['day'].replace(day_mapping)

    # Adjust Friday sales by adding 7000
    df.loc[df['day'] == 'Fri', 'sales_sum'] = df.loc[df['day'] == 'Fri', 'sales_sum'] + 7000
    df = df[df["cost_sum"] != 0]
    
    df_lg = df.copy()
    df_lg = df_lg.dropna()
    df_lg_plot = df_lg.copy()
    print("df_lg_plot.shape")
    print(df_lg_plot.shape)
    print(df_lg.tail(10))

    # Return cleaned dataframe
    return df_lg.copy(), df_lg_plot.copy()
#--------------------------------------------------------------------------

# (4) Logistic Regression Sales Forecasting Function
def forecast_sales_logistic(df_lg):
    import numpy as np
    # Assuming df_lg is already defined as in your provided code.
    from datetime import timedelta
  
    # Get the date 90 days ago from the latest date in the DataFrame.
    ninety_days_ago = df_lg['date'].max() - timedelta(days=90)

    # Filter the DataFrame to include only the last 90 days of data.
    df_lg = df_lg[df_lg['date'] >= ninety_days_ago]
    
    threshold = df_lg['sales_sum'].quantile(0.75)

    import joblib
    from sklearn.preprocessing import MinMaxScaler
    from datetime import timedelta

    # โหลดโมเดล
    model = joblib.load("/var/www/html/models/Logistic_Regression_Model.pkl")

    # เตรียม lag จาก training set
    lag = 7
    for i in range(1, lag + 1):
        df_lg[f'sales_sum_lag{i}'] = df_lg['sales_sum'].shift(i)
    df_train = df_lg.dropna().copy()

    features = [f'sales_sum_lag{i}' for i in range(1, lag + 1)] + [
        'sales_fs', 'sales_gb', 'ismarketday', 'isschoolday', 'holiday', 'V'
    ]

    df_x = df_train[features]
    scaler = MinMaxScaler()
    scaler.fit(df_x)

    # เตรียมข้อมูลสำหรับทำนายล่วงหน้า 7 วัน
    last_date = df_lg['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=7, freq='D')
    df_future = pd.DataFrame({'date': future_dates})

    # ฟังก์ชันสำหรับคำนวณค่าเฉลี่ยย้อนหลัง 3 เดือนของวันเดียวกัน
    def compute_historical_avg(df_lg, target_date):
        weekday = target_date.dayofweek
        past_3_months = df_lg[(df_lg['date'] < target_date) & (df_lg['date'] >= target_date - timedelta(days=90))]
        filtered = past_3_months[past_3_months['date'].dt.dayofweek == weekday]
        return filtered[['sales_fs', 'sales_gb', 'V']].mean()

    # ฟังก์ชันสำหรับคัดลอกค่าจากวันล่าสุดในอดีตที่ตรงกับวันในสัปดาห์
    def get_latest_matching_day_info(df_lg, target_date):
        weekday = target_date.dayofweek
        past_matching = df_lg[(df_lg['date'] < target_date) & (df_lg['date'].dt.dayofweek == weekday)]
        if not past_matching.empty:
            last_match = past_matching.sort_values('date').iloc[-1]
            return last_match[['ismarketday', 'isschoolday', 'holiday']]
        else:
            return pd.Series([0, 0, 0], index=['ismarketday', 'isschoolday', 'holiday'])

    # ค่า sales_sum ล่าสุด 7 วันจาก training set
    x_ = df_train['sales_sum'].iloc[-lag:].values.copy()
    p = np.zeros(len(df_future))

    # พยากรณ์ทีละวัน
    for t in range(len(df_future)):
        date_t = df_future.iloc[t]['date']
        avg_vals = compute_historical_avg(df_lg, date_t)
        cat_vals = get_latest_matching_day_info(df_lg, date_t)

        # ถ้ามี missing ใน avg ให้แทนด้วย 0
        avg_vals = avg_vals.fillna(0)

        x_input = x_.tolist() + avg_vals.tolist() + cat_vals.tolist()
        x_input_scaled = scaler.transform([x_input])
        p[t] = model.predict(x_input_scaled)[0]

        x_ = np.roll(x_, -1)
        x_[-1] = p[t]

    # สร้าง DataFrame ผลลัพธ์ 7 วันล่วงหน้า
    df_forecast_logistic = df_future.copy()
    df_forecast_logistic['predicted_sales'] = p
    df_forecast_logistic['day'] = df_forecast_logistic['date'].dt.strftime('%a')

    print(">>> df_forecast_logistic")
    print(threshold)
    print(df_forecast_logistic)

    return df_forecast_logistic
#--------------------------------------------------------------------------

# (4) get_google_trend using selenium--------------------------------------
# Set the download directory
download_dir = os.path.expanduser("~/Downloads")

# Setup WebDriver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_experimental_option("prefs", {"download.default_directory": download_dir})

def fetch_google_trends_selenium():
    """Fetch Google Trends using Selenium, but avoid re-downloading if today's CSV exists."""
    try:
        # Step 1: Check if today's file exists
        today_str = datetime.today().strftime("%Y%m%d")  # Example: "20250319"
        csv_files = [f for f in os.listdir(download_dir) if f.startswith(f"trending_TH_1d_{today_str}") and f.endswith(".csv")]
        
        if csv_files:
            latest_csv = max(csv_files, key=lambda f: os.path.getctime(os.path.join(download_dir, f)))
            print(f"Today's CSV already exists: {latest_csv}")
            df = pd.read_csv(os.path.join(download_dir, latest_csv))
            print("\nGoogle Trends CSV Data Loaded 🔹")
            print(df.head())  # Show first 5 rows
            return {"status": "success", "message": "CSV already exists", "trends": df.to_dict(orient="records")}

        # Step 2: Proceed to download if today's file is missing
        print("Downloading new Google Trends CSV...")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        # Open Google Trends
        url = "https://trends.google.co.th/trending?geo=TH"
        driver.get(url)
        time.sleep(5)  # Wait for page to load

        # Click "Export"
        export_button = driver.find_element(By.XPATH, "//button[contains(@aria-label, 'ส่งออก')]")
        driver.execute_script("arguments[0].click();", export_button)
        time.sleep(2)

        # Click "Download CSV"
        download_csv_button = driver.find_element(By.XPATH, "//span[contains(text(), 'ดาวน์โหลด CSV')]")
        driver.execute_script("arguments[0].click();", download_csv_button)
        time.sleep(5)

        driver.quit()  # Close the browser

        # Step 3: Find and load the newly downloaded file
        time.sleep(3)  # Ensure file is fully written before reading

        csv_files = [f for f in os.listdir(download_dir) if f.startswith(f"trending_TH_1d_{today_str}") and f.endswith(".csv")]
        if not csv_files:
            print("No CSV file found after download.")
            return {"status": "error", "message": "No CSV file found after download."}

        latest_csv = max(csv_files, key=lambda f: os.path.getctime(os.path.join(download_dir, f)))
        df = pd.read_csv(os.path.join(download_dir, latest_csv))

        print("\nNew Google Trends CSV Downloaded and Loaded ")
        print(df.head())  # Show first 5 rows

        return {"status": "success", "message": "New CSV downloaded", "trends": df.to_dict(orient="records")}

    except Exception as e:
        print(f"Error: {e}")
        return {"status": "error", "message": str(e)}

# (5) Get weekly summary data
def get_weekly_summary_data():
    df = get_sales_data()
    preprocess_data_prophet(df)

    global df_to_calculate
    if df_to_calculate is None or df_to_calculate.empty:
        return None

    last_date = df_to_calculate['date'].max()
    last_date = pd.to_datetime(last_date).normalize()

    start_of_week = last_date - pd.to_timedelta(last_date.weekday() + 1, unit='D')
    end_of_week = start_of_week + pd.Timedelta(days=6)

    df_weekly = df_to_calculate[
        (df_to_calculate['date'] >= start_of_week) & 
        (df_to_calculate['date'] <= end_of_week)
    ]
    
    print("df_to_calculate['date'] range:", df_to_calculate['date'].min(), "->", df_to_calculate['date'].max())
    print("last_date:", last_date)
    print("start_of_week:", start_of_week)
    print("end_of_week:", end_of_week)
    print("df_weekly rows:", df_weekly.shape[0])

    
    return {
        "weekly_cost": df_weekly['cost_sum'].sum(),
        "weekly_sales": df_weekly['sales_sum'].sum(),
        "weekly_profit": df_weekly['profit'].sum()
    }
#----------------------------------------------------------------------------------------

# 5. API Routes (return JSON)------------------------------------------------------------
# (5) Root Route to Check API Status
@app.route('/')
def home():
    return "Flask API is running! Use /forecast to get predictions.", 200

# (5) API Endpoint for Sales Forecasting
@app.route('/forecast', methods=['GET'])
def forecast():
    df = get_sales_data()
    df_prophet = preprocess_data_prophet(df)
    df_forecast = forecast_sales(df_prophet)
    return jsonify(df_forecast.to_dict(orient="records"))

# (5) Train_data_GET
@app.route('/train_data', methods=['GET'])
def get_train_data():
    df = get_sales_data()  # Fetch data from Google Sheets
    df_prophet = preprocess_data_prophet(df)  # Preprocess data
    train_data = df_prophet
    
        # Ensure `ds` is a datetime column before sorting
    train_data['ds'] = pd.to_datetime(train_data['ds'])
    
    return jsonify(train_data.to_dict(orient="records"))

# (5) Last upadated date
@app.route('/last_updated_date', methods=['GET'])
def last_updated_date():
    df = get_sales_data()
    df_prophet = preprocess_data_prophet(df)
    train_data = df_prophet

    train_data['ds'] = pd.to_datetime(train_data['ds'])
    last_date = train_data['ds'].max()

    return jsonify({
        "last_updated_date": last_date.strftime("%Y-%m-%d")  #format เป็น string
    })

# (5) Google trend selenium
@app.route('/google_trends_selenium', methods=['GET'])
def google_trends_selenium():
    trends = fetch_google_trends_selenium()
    if "trends" in trends:
        return jsonify({"trending_searches": trends["trends"]})
    else:
        return jsonify({"error": trends["message"]})

# (5) GET weekly summary
@app.route('/weekly_summary', methods=['GET'])
def weekly_summary():
    result = get_weekly_summary_data()
    if result is None:
        return jsonify({"error": "Data is not available"}), 500
    return jsonify(result)
    
# (5) Prophet summary status
@app.route('/summary_status', methods=['GET'])
def summary_status():
    df = get_sales_data()
    df_prophet = preprocess_data_prophet(df)
    df_forecast = forecast_sales(df_prophet)

    
    #**********************************EasySummaryTable**********************************
    # คำนวณค่า Percentile และค่าเฉลี่ยของยอดขายทั้งหมด
    sales_percentile_75 = df_forecast['predicted_sales'].quantile(0.75)
    sales_mean = df_forecast['predicted_sales'].mean()

    # วันนี้
    today = df_forecast[df_forecast['date'] == pd.to_datetime('today').normalize()]
    if not today.empty:
        A = today['predicted_sales'].values[0]
        A_label = "ขายดี" if A >= sales_percentile_75 else "ปกติ" if A >= sales_mean else "ท่าจะเงียบ" if A > 0 else "วันหยุด"
    else:
        A_label = "ไม่มีข้อมูล"

    # พรุ่งนี้
    tomorrow = df_forecast[df_forecast['date'] == (pd.to_datetime('today') + pd.Timedelta(days=1)).normalize()]
    if not tomorrow.empty:
        B = tomorrow['predicted_sales'].values[0]
        B_label = "ขายดี" if B >= sales_percentile_75 else "ปกติ" if B >= sales_mean else "ท่าจะเงียบ" if B > 0 else "วันหยุด"
    else:
        B_label = "ไม่มีข้อมูล"

    # สัปดาห์นี้ (เฉพาะวันอาทิตย์ถึงวันเสาร์ของสัปดาห์นี้ และไม่รวมค่าที่เป็น 0)
    this_week = df_forecast[(df_forecast['date'] >= pd.to_datetime('today') - pd.to_timedelta(pd.to_datetime('today').weekday(), unit='D')) & 
                        (df_forecast['date'] <= pd.to_datetime('today') + pd.Timedelta(days=6 - pd.to_datetime('today').weekday())) &
                        (df_forecast['predicted_sales'] > 0)]
    if not this_week.empty:
        C = this_week['predicted_sales'].mean()
        C_label = "ขายดี" if C >= sales_percentile_75 else "ปกติ" if C >= sales_mean else "ท่าจะเงียบ" if C > 0 else "วันหยุด"
    else:
        C_label = "ไม่มีข้อมูล"

    # สัปดาห์หน้า (เฉพาะวันอาทิตย์ถึงวันเสาร์ของสัปดาห์หน้า และไม่รวมค่าที่เป็น 0)
    next_sunday = pd.to_datetime('today') + pd.Timedelta(days=(7 - pd.to_datetime('today').weekday()))
    next_saturday = next_sunday + pd.Timedelta(days=6)
    next_week = df_forecast[(df_forecast['date'] >= next_sunday) & 
                        (df_forecast['date'] <= next_saturday) & 
                        (df_forecast['predicted_sales'] > 0)]
    if not next_week.empty:
        D = next_week['predicted_sales'].mean()
        D_label = "ขายดี" if D >= sales_percentile_75 else "ปกติ" if D >= sales_mean else "ท่าจะเงียบ" if D > 0 else "วันหยุด"
    else:
        D_label = "ไม่มีข้อมูล"

    # สร้าง DataFrame สำหรับแสดงใน Dashboard
    df_summary = pd.DataFrame({
        "หมวดหมู่": ["วันนี้", "พรุ่งนี้", "สัปดาห์นี้", "สัปดาห์หน้า"],
        "สถานะ": [A_label, B_label, C_label, D_label]
    })

    
    def label(value):
        if value >= sales_percentile_75:
            return "ขายดี"
        elif value >= sales_mean:
            return "ปกติ"
        elif value > 0:
            return "ท่าจะเงียบ"
        else:
            return "วันหยุด"

    return jsonify({
        "today": label(today['predicted_sales'].values[0]) if not today.empty else "ไม่มีข้อมูล",
        "tomorrow": label(tomorrow['predicted_sales'].values[0]) if not tomorrow.empty else "ไม่มีข้อมูล",
        "this_week": label(this_week['predicted_sales'].mean()) if not this_week.empty else "ไม่มีข้อมูล",
        "next_week": label(next_week['predicted_sales'].mean()) if not next_week.empty else "ไม่มีข้อมูล"
    })
#--------------------------------------------------------------------------
    
# (5)  Linear Regression Summary Status
def summary_status_linear(df_forecast_linear):
    sales_percentile_75 = df_forecast_linear['predicted_sales'].quantile(0.75)
    sales_mean = df_forecast_linear['predicted_sales'].mean()

    today = df_forecast_linear[df_forecast_linear['date'] == pd.to_datetime('today').normalize()]
    tomorrow = df_forecast_linear[df_forecast_linear['date'] == (pd.to_datetime('today') + pd.Timedelta(days=1)).normalize()]

    def label(value):
        if value >= sales_percentile_75:
            return "ขายดี"
        elif value >= sales_mean:
            return "ปกติ"
        elif value > 0:
            return "ท่าจะเงียบ"
        else:
            return "วันหยุด"

    this_week = df_forecast_linear[(df_forecast_linear['date'] >= pd.to_datetime('today') - pd.to_timedelta(pd.to_datetime('today').weekday(), unit='D')) &
                            (df_forecast_linear['date'] <= pd.to_datetime('today') + pd.Timedelta(days=6 - pd.to_datetime('today').weekday())) &
                            (df_forecast_linear['predicted_sales'] > 0)]

    next_sunday = pd.to_datetime('today') + pd.Timedelta(days=(7 - pd.to_datetime('today').weekday()))
    next_saturday = next_sunday + pd.Timedelta(days=6)
    next_week = df_forecast_linear[(df_forecast_linear['date'] >= next_sunday) & 
                             (df_forecast_linear['date'] <= next_saturday) & 
                             (df_forecast_linear['predicted_sales'] > 0)]

    return {
        "today": label(today['predicted_sales'].values[0]) if not today.empty else "ไม่มีข้อมูล",
        "tomorrow": label(tomorrow['predicted_sales'].values[0]) if not tomorrow.empty else "ไม่มีข้อมูล",
        "this_week": label(this_week['predicted_sales'].mean()) if not this_week.empty else "ไม่มีข้อมูล",
        "next_week": label(next_week['predicted_sales'].mean()) if not next_week.empty else "ไม่มีข้อมูล"
    }
#---------------------------------------------------------------------------


# (5) Logistic Regression Summary Status
def summary_status_logistic(df_forecast_logistic):
    today = pd.to_datetime('today').normalize()
    tomorrow = (pd.to_datetime('today') + pd.Timedelta(days=1)).normalize()

    # หาวันอาทิตย์-เสาร์ของสัปดาห์นี้
    start_this_week = today - pd.to_timedelta(today.weekday(), unit='D')
    end_this_week = start_this_week + pd.Timedelta(days=6)

    # หาวันอาทิตย์-เสาร์ของสัปดาห์หน้า
    start_next_week = end_this_week + pd.Timedelta(days=1)
    end_next_week = start_next_week + pd.Timedelta(days=6)

    # ตัวช่วยแปลงผล
    def label(binary_value):
        return "ขายดี" if binary_value >= 0.6 else "ปกติ"

    # กรองข้อมูล
    today_val = df_forecast_logistic[df_forecast_logistic['date'] == today]
    tomorrow_val = df_forecast_logistic[df_forecast_logistic['date'] == tomorrow]
    this_week_vals = df_forecast_logistic[(df_forecast_logistic['date'] >= start_this_week) & (df_forecast_logistic['date'] <= end_this_week)]
    next_week_vals = df_forecast_logistic[(df_forecast_logistic['date'] >= start_next_week) & (df_forecast_logistic['date'] <= end_next_week)]

    return {
        "today": "ขายดี" if not today_val.empty and today_val['predicted_sales'].values[0] == 1 else "ปกติ",
        "tomorrow": "ขายดี" if not tomorrow_val.empty and tomorrow_val['predicted_sales'].values[0] == 1 else "ปกติ",
        "this_week": label(this_week_vals['predicted_sales'].mean()) if not this_week_vals.empty else "ไม่มีข้อมูล",
        "next_week": label(next_week_vals['predicted_sales'].mean()) if not next_week_vals.empty else "ไม่มีข้อมูล"
    }

#----------------------------------------------------------------------------

# (5) Prophet plot_all_sales_vs_forecast 
@app.route('/plot/sales_vs_forecast')
def plot_all_sales_vs_forecast():
    df = get_sales_data()
    df_prophet = preprocess_data_prophet(df)
    df_forecast = forecast_sales(df_prophet)

    # เตรียมข้อมูล
    df_train = df_prophet.copy()
    df_train = df_train[df_train['y'] > 0]

    avg_forecast = df_forecast['predicted_sales'].mean()
    
    print(df_forecast['predicted_sales'])
    print(df_forecast['predicted_sales'].mean())
    
    fig = px.line(
        x=df_train['ds'],
        y=df_train['y'],
        line_shape="linear",
        #title="All Sales vs 1 Month Average Forecast",
        labels={"x": "วันที่", "y": "ยอดขาย (บาท)"}
    )

    fig.update_traces(line=dict(color='blue'), name="ยอดขายจริง", showlegend=True)

    fig.add_hline(
        y=avg_forecast,
        line_dash="dash",
        line_color="red",
        annotation_text="ค่าทำนายเฉลี่ย",
        annotation_position="top right",
        showlegend=False
    )
    fig.add_trace(
    go.Scatter(
        x=[None], y=[None], mode='lines',
        line=dict(color='red', dash='dash'),
        name='ค่าทำนายเฉลี่ย'
        )
    )

    #fig.update_traces(line=dict(color='red'), name="ค่าทำนายเฉลี่ย")
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))

    all_sales_graph_html = pio.to_html(fig, full_html=False)

    return Markup(all_sales_graph_html)
#------------------------------------------------------------------------

# (5) Linear Regression: plot_all_sales_vs_forecast
@app.route('/plot/sales_vs_forecast_linear')
def plot_all_sales_vs_forecast_linear():
    df = get_sales_data()
    df_lr, df_lr_plot = preprocess_data_linear(df)
    df_forecast_linear = forecast_sales_linear(df_lr)

    import numpy as np

    df_forecast_linear['predicted_sales'] = df_forecast_linear['predicted_sales'].apply(
        lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x
    )

    # เตรียมข้อมูลฝั่งจริง (ย้อนหลัง 90 วัน)
    df_train = df_lr_plot.copy()
    df_train = df_train[df_train['sales_sum'] > 0]
    df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')
    #df_train = df_train[df_train['date'] >= df_train['date'].max() - pd.Timedelta(days=90)]
    df_train['day'] = df_train['date'].dt.strftime('%a')

    avg_forecast = df_forecast_linear['predicted_sales'].mean()
    print("avg_forecast", avg_forecast)

    fig = px.line(
        x=df_train['date'],
        y=df_train['sales_sum'],
        line_shape="linear",
        labels={"x": "วันที่", "y": "ยอดขาย (บาท)"}
    )

    fig.update_traces(line=dict(color='blue'), name="ยอดขายจริง", showlegend=True)

    fig.add_hline(
        y=avg_forecast,
        line_dash="dash",
        line_color="red",
        annotation_text="ค่าทำนายเฉลี่ย",
        annotation_position="top right",
        showlegend=False
    )
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode='lines',
            line=dict(color='red', dash='dash'),
            name='ค่าทำนายเฉลี่ย'
        )
    )

    fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))

    all_sales_graph_html = pio.to_html(fig, full_html=False)
    return Markup(all_sales_graph_html)
#-------------------------------------------------------------------------

# (5) Logistic Regression plot_all_sales_vs_forecast
@app.route('/plot/sales_vs_forecast_logistic')
def plot_all_sales_vs_forecast_logistic():
    from plotly import graph_objects as go

    df = get_sales_data()
    df_lg, df_lg_plot = preprocess_data_logistic(df)
    df_forecast_logistic = forecast_sales_logistic(df_lg)

    # เตรียมข้อมูลยอดขายจริง
    df_train = df_lg_plot.copy()
    df_train = df_train[df_train['sales_sum'] > 0]
    df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')

    # คำนวณ threshold
    threshold = df_train['sales_sum'].quantile(0.75)

    # เตรียม figure
    fig = go.Figure()

    # เส้นยอดขายจริง (สีฟ้า)
    fig.add_trace(go.Scatter(
        x=df_train['date'],
        y=df_train['sales_sum'],
        mode='lines',
        name='ยอดขายจริง',
        line=dict(color='blue')
    ))

    # เส้น threshold แนวนอน (สีแดง)
    fig.add_hline(
        y=threshold,
        line_color="red",
        line_dash="dash"
    )

    # เพิ่ม annotation ด้านซ้ายล่างของเส้น threshold
    fig.add_annotation(
        xref="paper", y=threshold,
        x=0,  # ซ้ายสุด
        text="เกินเส้นนี้คือขายดี",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(color="red"),
        bgcolor="rgba(255,255,255,0.6)",  # พื้นหลังขาวโปร่ง
        bordercolor="red",
        borderwidth=1,
        borderpad=4
    )

    # เตรียมข้อความ "ขายดี" วันละบรรทัดที่มุมขวาล่าง
    df_forecast_logistic['date'] = pd.to_datetime(df_forecast_logistic['date'])
    hot_sales = df_forecast_logistic[df_forecast_logistic['predicted_sales'] == 1]

    hot_texts = []
    for _, row in hot_sales.iterrows():
        day = row['date'].strftime("%d")
        month = row['date'].strftime("%m")
        hot_texts.append(f"วันที่ {day} เดือน {month} : ขายดี !")

    if hot_texts:
        summary_text = "<br>".join(hot_texts)
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0, y=0,
            xanchor="left", yanchor="bottom",
            text=summary_text,
            showarrow=False,
            font=dict(color="green", size=12),
            align="right",
            bgcolor="rgba(255,255,255,0.6)",
            bordercolor="green",
            borderwidth=1,
            borderpad=4
        )

    # กำหนดช่วงแกน x ครอบคลุมอดีตถึงอนาคต
    x_min = df_train['date'].min()
    x_max = df_forecast_logistic['date'].max()

    fig.update_layout(
        xaxis=dict(range=[x_min, x_max]),
        yaxis_title="ยอดขาย (บาท)",
        xaxis_title="วันที่",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    all_sales_graph_html = pio.to_html(fig, full_html=False)
    return Markup(all_sales_graph_html)

#------------------------------------------------------------------------

# (5) Prophet Day of week vs forecast
@app.route('/plot/7_day_forecast')
def plot_7_day_forecast():
    df = get_sales_data()
    df_prophet = preprocess_data_prophet(df)
    df_forecast = forecast_sales(df_prophet)

    df_train = df_prophet.copy()
    df_train['day'] = df_train['ds'].dt.strftime('%a')
    avg_sales_per_day = df_train.groupby('day')['y'].mean().reset_index()
    avg_sales_per_day.columns = ['day', 'avg_sales']

    df_forecast['day'] = df_forecast['date'].dt.strftime('%a')
    df_forecast['day_with_date'] = df_forecast['date'].dt.strftime('%a [%Y-%m-%d]')

    df_forecast = df_forecast.merge(avg_sales_per_day, on='day', how='left')
    df_forecast_DayOfWeek = df_forecast.tail(7)

    # กรองเฉพาะค่าที่ไม่เป็นศูนย์
    df_avg_only = df_forecast_DayOfWeek[df_forecast_DayOfWeek['avg_sales'] > 0]
    df_predicted_only = df_forecast_DayOfWeek[df_forecast_DayOfWeek['predicted_sales'] > 0]

    # วาดกราฟด้วย go.Figure
    fig = go.Figure()

    # เส้น avg_sales (blue)
    fig.add_trace(go.Scatter(
        x=df_avg_only['day_with_date'],
        y=df_avg_only['avg_sales'],
        mode='lines+markers',
        name='avg_sales',
        line=dict(color='blue')
    ))

    # เส้น predicted_sales (red)
    fig.add_trace(go.Scatter(
        x=df_predicted_only['day_with_date'],
        y=df_predicted_only['predicted_sales'],
        mode='lines+markers',
        name='predicted_sales',
        line=dict(color='red')
    ))

    fig.update_xaxes(title="วันที่")
    fig.update_yaxes(title="ยอดขาย (บาท)", range=[0, max(
        df_avg_only['avg_sales'].max() if not df_avg_only.empty else 0,
        df_predicted_only['predicted_sales'].max() if not df_predicted_only.empty else 0
    )])
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))

    forecast_graph_html = pio.to_html(fig, full_html=False)
    return Markup(forecast_graph_html)

#-------------------------------------------------------------------

# (5) Linear Regression: Day of week vs forecast
@app.route('/plot/7_day_forecast_linear')
def plot_7_day_forecast_linear():
    df = get_sales_data()
    df_lr, df_lr_plot = preprocess_data_linear(df)
    df_forecast_linear = forecast_sales_linear(df_lr)

    import numpy as np
    import plotly.graph_objects as go

    df_forecast_linear['predicted_sales'] = df_forecast_linear['predicted_sales'].apply(
        lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x
    )

    # สร้าง avg_sales ต่อวันย้อนหลัง 90 วัน
    df_train = df_lr_plot.copy()
    df_train = df_train[df_train['sales_sum'] > 0]
    df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')
    df_train = df_train[df_train['date'] >= df_train['date'].max() - pd.Timedelta(days=90)]
    df_train['day'] = df_train['date'].dt.strftime('%a')
    avg_sales_per_day = df_train.groupby('day')['sales_sum'].mean().reset_index()
    avg_sales_per_day.columns = ['day', 'avg_sales']

    # เตรียม df สำหรับกราฟ
    df_forecast_linear['day'] = df_forecast_linear['date'].dt.strftime('%a')
    df_forecast_linear['day_with_date'] = df_forecast_linear['date'].dt.strftime('%a [%Y-%m-%d]')
    df_forecast_linear = df_forecast_linear.merge(avg_sales_per_day, on='day', how='left')
    df_forecast_DayOfWeek = df_forecast_linear.tail(7)

    # แยกข้อมูลออกตามเงื่อนไข
    df_predicted_only = df_forecast_DayOfWeek[df_forecast_DayOfWeek['predicted_sales'] > 0]

    # สร้างกราฟ
    fig = go.Figure()

    # เส้น avg_sales (ฟ้า)
    fig.add_trace(go.Scatter(
        x=df_forecast_DayOfWeek['day_with_date'],
        y=df_forecast_DayOfWeek['avg_sales'],
        mode='lines+markers',
        name='avg_sales',
        line=dict(color='blue')
    ))

    # เส้น predicted_sales (แดง) — เฉพาะที่ค่า > 0
    fig.add_trace(go.Scatter(
        x=df_predicted_only['day_with_date'],
        y=df_predicted_only['predicted_sales'],
        mode='lines+markers',
        name='predicted_sales',
        line=dict(color='red')
    ))

    fig.update_xaxes(title="วันที่")
    fig.update_yaxes(title="ยอดขาย (บาท)", range=[0, max(
        df_forecast_DayOfWeek['avg_sales'].max(),
        df_predicted_only['predicted_sales'].max() if not df_predicted_only.empty else 0
    )])
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))

    forecast_graph_html = pio.to_html(fig, full_html=False)
    return Markup(forecast_graph_html)
#-------------------------------------------------------------------------

# (5) Logistic Regression plot_7_days_vs_forecast
@app.route('/plot/7_day_forecast_logistic')
def plot_7_day_forecast_logistic():
    from plotly import graph_objects as go

    df = get_sales_data()
    df_lg, df_lg_plot = preprocess_data_logistic(df)
    df_forecast_logistic = forecast_sales_logistic(df_lg)

    # เตรียมข้อมูลย้อนหลัง 90 วัน
    df_train = df_lg_plot.copy()
    df_train = df_train[df_train['sales_sum'] > 0]
    df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')
    df_train = df_train[df_train['date'] >= df_train['date'].max() - pd.Timedelta(days=90)]
    df_train['day'] = df_train['date'].dt.strftime('%a')
    avg_sales_per_day = df_train.groupby('day')['sales_sum'].mean().reset_index()
    avg_sales_per_day.columns = ['day', 'avg_sales']

    # เตรียมข้อมูลพยากรณ์
    df_forecast_logistic['day'] = df_forecast_logistic['date'].dt.strftime('%a')
    df_forecast_logistic = df_forecast_logistic.merge(avg_sales_per_day, on='day', how='left')
    df_forecast_DayOfWeek = df_forecast_logistic.tail(7)

    # คำนวณ threshold จาก avg_sales
    #threshold = avg_sales_per_day['avg_sales'].quantile(0.75)
    threshold = df_train['sales_sum'].quantile(0.75)
    
    # สร้างกราฟ
    fig = go.Figure()

    # เส้นยอดขายเฉลี่ยย้อนหลัง (สีฟ้า)
    fig.add_scatter(
        x=df_forecast_DayOfWeek['date'],
        y=df_forecast_DayOfWeek['avg_sales'],
        mode='lines+markers',
        name='ยอดขายเฉลี่ย',
        line=dict(color='blue', width=3),
        marker=dict(size=10)
    )

    # เส้นคูณเฉลี่ย (สีแดง)
    fig.add_scatter(
        x=df_forecast_DayOfWeek['date'],
        y=df_forecast_DayOfWeek['predicted_sales'] * df_forecast_DayOfWeek['avg_sales'],
        mode='lines+markers',
        name='ขายดี',
        line=dict(color='red', width=3, dash='dot'),
        opacity=0.6,
        marker=dict(size=8)
    )

    # เส้น threshold แนวนอน
    fig.add_hline(
        y=threshold,
        line_color="red",
        line_dash="dash",
        annotation_text="เส้นขายดี",
        annotation_position="bottom left",
        annotation_font_color="red"
    )

    # ลูกศรขึ้น/ลงตาม predicted_sales
    for _, row in df_forecast_DayOfWeek.iterrows():
        fig.add_annotation(
        x=row['date'],
        y=row['avg_sales'],
        text="↑" if row['predicted_sales'] == 1 else "↓",
        showarrow=False,
        font=dict(
            color="green" if row['predicted_sales'] == 1 else "orange",
            size=20
        ),
        yshift=15 if row['predicted_sales'] == 1 else -15
    )


    fig.update_layout(
        xaxis_title="วันที่",
        yaxis_title="ยอดขาย (บาท)",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    forecast_graph_html = pio.to_html(fig, full_html=False)
    return Markup(forecast_graph_html)

#-------------------------------------------------------------------------

# (5) 10 Top Google Trend
@app.route('/google_trends_summary')
def google_trends_summary():
    trends = fetch_google_trends_selenium()
    if "trends" in trends and len(trends["trends"]) > 0:
        top_10_trends = [
            {
                "name": item.get("มาแรง", "ไม่มีชื่อ"),
                "volume": item.get("ปริมาณการค้นหา", "ไม่มีข้อมูล")
            }
            for item in trends["trends"][:10]
        ]
        return jsonify({"top_10_trends": top_10_trends})
    else:
        return jsonify({"top_10_trends": []})
#------------------------------------------------------------------------------

# (5) Prophet 3Month31Day vs forecast 
@app.route('/plot/sales_trend_1_31')
def plot_sales_trend_1_31():
    df = get_sales_data()
    df_prophet = preprocess_data_prophet(df)
    df_forecast = forecast_sales(df_prophet)

    df_train = df_prophet.copy()

    # ค่าเฉลี่ยย้อนหลัง 3 เดือน
    df_train['day_of_month'] = df_train['ds'].dt.day
    avg_sales_per_day_of_month = df_train.groupby('day_of_month')['y'].mean().reset_index()
    avg_sales_per_day_of_month.columns = ['day_of_month', 'avg_sales']

    # ค่าพยากรณ์ 30 วัน
    df_forecast['day_of_month'] = df_forecast['date'].dt.day
    df_forecast['day_with_date'] = df_forecast['day_of_month'].astype(str) + "[" + df_forecast['date'].dt.strftime('%Y-%m-%d') + "]"
    forecast_per_day_of_month = df_forecast.groupby(['day_of_month', 'day_with_date'])['predicted_sales'].mean().reset_index()

    # กรองค่าที่ไม่เป็น 0
    df_avg = avg_sales_per_day_of_month[avg_sales_per_day_of_month['avg_sales'] > 0]
    df_pred = forecast_per_day_of_month[forecast_per_day_of_month['predicted_sales'] > 0]

    # สร้างกราฟด้วย graph_objects
    fig = go.Figure()

    # เส้น avg_sales (ฟ้า)
    fig.add_trace(go.Scatter(
        x=df_avg['day_of_month'],
        y=df_avg['avg_sales'],
        mode='lines+markers',
        name='ยอดขายเฉลี่ยย้อนหลัง',
        line=dict(color='blue')
    ))

    # เส้น predicted_sales (แดง)
    fig.add_trace(go.Scatter(
        x=df_pred['day_of_month'],
        y=df_pred['predicted_sales'],
        mode='lines+markers',
        name='Forecasted Sales',
        line=dict(color='red'),
        hovertext=df_pred['day_with_date'],
        hoverinfo='text+y'
    ))

    fig.update_xaxes(title="Date (1–31)")
    fig.update_yaxes(title="ยอดขาย (บาท)", range=[0, max(df_avg['avg_sales'].max(), df_pred['predicted_sales'].max())])
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))

    thirthyone_linear_graph_html = pio.to_html(fig, full_html=False)
    return Markup(thirthyone_linear_graph_html)

#--------------------------------------------------------------------

# (5) Linear Regression: 3Month31Day vs forecast
@app.route('/plot/sales_trend_1_31_linear')
def plot_sales_trend_1_31_linear():
    df = get_sales_data()
    df_lr, df_lr_plot = preprocess_data_linear(df)
    df_forecast_linear = forecast_sales_linear(df_lr)

    import numpy as np
    import plotly.graph_objects as go

    df_forecast_linear['predicted_sales'] = df_forecast_linear['predicted_sales'].apply(
        lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x
    )

    # เตรียมข้อมูล avg_sales
    df_train = df_lr_plot.copy()
    df_train = df_train[df_train['sales_sum'] > 0]
    df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')
    df_train = df_train[df_train['date'] >= df_train['date'].max() - pd.Timedelta(days=90)]
    df_train['day_of_month'] = df_train['date'].dt.day
    avg_sales_per_day_of_month = df_train.groupby('day_of_month')['sales_sum'].mean().reset_index()
    avg_sales_per_day_of_month.columns = ['day_of_month', 'avg_sales']

    # เตรียม forecast
    df_forecast_linear['day_of_month'] = df_forecast_linear['date'].dt.day
    df_forecast_linear['day_with_date'] = df_forecast_linear['day_of_month'].astype(str) + "[" + df_forecast_linear['date'].dt.strftime('%Y-%m-%d') + "]"
    forecast_per_day_of_month = df_forecast_linear.groupby(['day_of_month', 'day_with_date'])['predicted_sales'].mean().reset_index()

    # กรอง avg_sales != 0
    df_avg = avg_sales_per_day_of_month[avg_sales_per_day_of_month['avg_sales'] > 0]

    # กรอง predicted_sales != 0
    df_pred = forecast_per_day_of_month[forecast_per_day_of_month['predicted_sales'] > 0]

    # สร้างกราฟ
    fig = go.Figure()

    # เส้นสีน้ำเงิน (ย้อนหลัง)
    fig.add_trace(go.Scatter(
        x=df_avg['day_of_month'],
        y=df_avg['avg_sales'],
        mode='lines+markers',
        name='ยอดขายเฉลี่ยย้อนหลัง',
        line=dict(color='blue')
    ))

    # เส้นสีแดง (ทำนายล่วงหน้า)
    fig.add_trace(go.Scatter(
        x=df_pred['day_of_month'],
        y=df_pred['predicted_sales'],
        mode='lines+markers',
        name='Forecasted Sales',
        line=dict(color='red'),
        hovertext=df_pred['day_with_date'],
        hoverinfo='text+y'
    ))

    fig.update_xaxes(title="Date (1–31)")
    fig.update_yaxes(title="ยอดขาย (บาท)", range=[0, max(df_avg['avg_sales'].max(), df_pred['predicted_sales'].max())])
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))

    thirthyone_linear_graph_html = pio.to_html(fig, full_html=False)
    return Markup(thirthyone_linear_graph_html)
#------------------------------------------------------------------

# (5) Logistic Regression: 3Month31Day vs forecast
@app.route('/plot/sales_trend_1_31_logistic')
def plot_sales_trend_1_31_logistic():
    from plotly import graph_objects as go

    df = get_sales_data()
    df_lg, df_lg_plot = preprocess_data_logistic(df)
    df_forecast_logistic = forecast_sales_logistic(df_lg)

    # ข้อมูลย้อนหลัง 3 เดือน
    df_train = df_lg_plot[df_lg_plot['sales_sum'] > 0].copy()
    df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')
    df_train = df_train[df_train['date'] >= df_train['date'].max() - pd.Timedelta(days=90)]

    df_train['day_of_month'] = df_train['date'].dt.day
    avg_sales_per_day = df_train.groupby('day_of_month')['sales_sum'].mean().reset_index()
    avg_sales_per_day.columns = ['day_of_month', 'avg_sales']
    #threshold = avg_sales_per_day['avg_sales'].quantile(0.75)
    threshold = df_lg_plot['sales_sum'].quantile(0.75)

    # ข้อมูลทำนาย
    df_forecast_logistic['day_of_month'] = df_forecast_logistic['date'].dt.day
    df_forecast_logistic['day_with_date'] = df_forecast_logistic['date'].dt.strftime('%Y-%m-%d')
    forecast = df_forecast_logistic[['day_of_month', 'day_with_date', 'predicted_sales']]

    # สร้างกราฟ
    fig = go.Figure()

    # เส้นฟ้า: ยอดขายเฉลี่ย
    fig.add_trace(go.Scatter(
        x=avg_sales_per_day['day_of_month'],
        y=avg_sales_per_day['avg_sales'],
        mode='lines+markers',
        name='ยอดขายเฉลี่ย',
        line=dict(color='blue')
    ))

    # เส้นแดงแนวนอน: threshold
    fig.add_hline(
        y=threshold,
        line_color="red",
        line_dash="dash",
        annotation_text="Threshold",
        annotation_position="bottom right",
        annotation_font_color="red"
    )

    # แสดงลูกศรออกจากจุด (↑ หรือ ↓)
    for _, row in forecast.iterrows():
        day = row['day_of_month']
        pred = row['predicted_sales']
        match = avg_sales_per_day[avg_sales_per_day['day_of_month'] == day]
        if not match.empty:
            y_val = match['avg_sales'].values[0]
            # กำหนดลูกศรให้แสดง "ออกจากจุด" เล็กน้อย
            arrow_symbol = "↑" if pred == 1 else "↓"
            color = "green" if pred == 1 else "orange"
            fig.add_trace(go.Scatter(
                x=[day],
                y=[y_val + 500 if pred == 1 else y_val - 500],
                mode="text",
                text=[arrow_symbol],
                textfont=dict(size=24, color=color, family="Arial Black"),  # เพิ่มขนาดและความหนา
                showlegend=False
            ))

    fig.update_layout(
        xaxis_title="วันที่ (1–31)",
        yaxis_title="ยอดขาย (บาท)",
        yaxis=dict(range=[0, avg_sales_per_day['avg_sales'].max() + 5000]),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return Markup(pio.to_html(fig, full_html=False))
#------------------------------------------------------------------

# (5) Prophet Monthly vs forecast 
@app.route('/plot/monthly_sales_vs_forecast')
def plot_monthly_sales_vs_forecast():
    df = get_sales_data()
    df_prophet = preprocess_data_prophet(df)
    df_forecast = forecast_sales(df_prophet)

    df_train = df_prophet.copy()
    df_train['month'] = df_train['ds'].dt.strftime('%b')
    avg_sales_per_month = df_train.groupby('month')['y'].mean().reset_index()
    avg_sales_per_month.columns = ['month', 'avg_sales']

    # 🔺 กรองค่า 0 หรือ NaN
    avg_sales_per_month = avg_sales_per_month[avg_sales_per_month['avg_sales'] > 0].dropna()

    avg_forecast = df_forecast[df_forecast['predicted_sales'] > 0]['predicted_sales'].mean()

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    avg_sales_per_month['month'] = pd.Categorical(
        avg_sales_per_month['month'],
        categories=month_order,
        ordered=True
    )
    avg_sales_per_month = avg_sales_per_month.sort_values('month')

    # ใช้ go.Figure แทน px.line เพื่อควบคุมข้อผิดพลาด
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=avg_sales_per_month['month'],
        y=avg_sales_per_month['avg_sales'],
        mode='lines+markers',
        name='ยอดขายเฉลี่ยย้อนหลัง',
        line=dict(color='blue')
    ))

    # เส้นคาดการณ์เฉลี่ย
    fig.add_hline(
        y=avg_forecast,
        line_dash="dash",
        line_color="red",
        annotation_text="Forecast Avg",
        annotation_position="top right"
    )

    # กล่องข้อความสรุปค่า
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.05,
        text=f"AVG 1 Month Forecast: {avg_forecast:.2f} THB",
        showarrow=False,
        font=dict(size=12, color="red"),
        bgcolor="white",
        bordercolor="red",
        borderwidth=1
    )

    fig.update_xaxes(title="Month (Jan–Dec)")
    fig.update_yaxes(title="ยอดขาย (บาท)", range=[0, max(avg_sales_per_month['avg_sales'].max(), avg_forecast)])
    fig.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))

    monthly_graph_html = pio.to_html(fig, full_html=False)
    return Markup(monthly_graph_html)

#----------------------------------------------------------------------

# (5) Linear Regression: Monthly vs forecast
@app.route('/plot/monthly_sales_vs_forecast_linear')
def plot_monthly_sales_vs_forecast_linear():
    df = get_sales_data()
    df_lr, df_lr_plot = preprocess_data_linear(df)
    df_forecast_linear = forecast_sales_linear(df_lr)

    import numpy as np

    df_forecast_linear['predicted_sales'] = df_forecast_linear['predicted_sales'].apply(
        lambda x: x[0] if isinstance(x, (list, np.ndarray)) else x
    )

    df_train = df_lr_plot.copy()
    df_train = df_train[df_train['sales_sum'] > 0]
    df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')
    df_train['month'] = df_train['date'].dt.strftime('%b')
    avg_sales_per_month = df_train.groupby('month')['sales_sum'].mean().reset_index()
    avg_sales_per_month.columns = ['month', 'avg_sales']

    avg_forecast = df_forecast_linear[df_forecast_linear['predicted_sales'] > 0]['predicted_sales'].mean()

    # จัดลำดับเดือน
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    avg_sales_per_month['month'] = pd.Categorical(avg_sales_per_month['month'], categories=month_order, ordered=True)
    avg_sales_per_month = avg_sales_per_month.sort_values('month')

    # สร้างกราฟ
    fig_monthly = px.line(
        avg_sales_per_month, 
        x='month', 
        y='avg_sales', 
        markers=True
    )

    fig_monthly.add_hline(
        y=avg_forecast, 
        line_dash="dash", 
        line_color="red", 
        annotation_text="Forecast Avg", 
        annotation_position="top right"
    )

    fig_monthly.update_traces(line=dict(color='blue'))
    fig_monthly.update_xaxes(title="Month (Jan-Dec)")
    fig_monthly.update_yaxes(title="ยอดขาย (บาท)")

    fig_monthly.add_annotation(
        xref="paper", yref="paper",
        x=0.95, y=0.95,
        text=f"AVG 1 Month Forecast: {float(avg_forecast):.2f} THB",
        showarrow=False,
        font=dict(size=12, color="red"),
        bgcolor="white",
        bordercolor="red",
        borderwidth=1
    )

    max_value = max(avg_sales_per_month['avg_sales'].max(), avg_forecast)
    fig_monthly.update_yaxes(range=[0, max_value])
    fig_monthly.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))

    monthly_graph_html = pio.to_html(fig_monthly, full_html=False)
    return Markup(monthly_graph_html)
#---------------------------------------------------------------------


# (5) Logistic Regression: Monthly vs forecast  
@app.route('/plot/monthly_sales_vs_forecast_logistic')
def plot_monthly_sales_vs_forecast_logistic():
    df = get_sales_data()
    df_lg, df_lg_plot = preprocess_data_logistic(df)
    df_forecast_logistic = forecast_sales_logistic(df_lg)

    # เตรียมข้อมูลย้อนหลัง
    df_train = df_lg_plot.copy()
    df_train = df_train[df_train['sales_sum'] > 0]
    df_train['date'] = pd.to_datetime(df_train['date'], errors='coerce')
    df_train['month'] = df_train['date'].dt.strftime('%b')
    avg_sales_per_month = df_train.groupby('month')['sales_sum'].mean().reset_index()
    avg_sales_per_month.columns = ['month', 'avg_sales']

    # คำนวณ threshold
    threshold = df_lg_plot['sales_sum'].quantile(0.75)

    # จัดลำดับเดือน
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    avg_sales_per_month['month'] = pd.Categorical(avg_sales_per_month['month'], categories=month_order, ordered=True)
    avg_sales_per_month = avg_sales_per_month.sort_values('month')

    # กราฟยอดขายเฉลี่ย
    fig_monthly = px.line(
        avg_sales_per_month,
        x='month',
        y='avg_sales',
        markers=True
    )
    fig_monthly.update_traces(line=dict(color='blue'), name='ยอดขายเฉลี่ยย้อนหลัง')

    # เส้นประ threshold สีแดง
    fig_monthly.add_hline(
        y=threshold,
        line_color="red",
        line_dash="dash"
    )

    # กล่องข้อความ "เกินเส้นนี้คือขายดี"
    fig_monthly.add_annotation(
        xref="paper", yref="y",
        x=0.01, y=threshold,
        text="เกินเส้นนี้คือขายดี",
        showarrow=False,
        font=dict(size=12, color="red"),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="red",
        borderwidth=1
    )

    # เตรียมข้อความกล่องมุมซ้ายล่าง
    df_forecast_logistic['date'] = pd.to_datetime(df_forecast_logistic['date'], errors='coerce')
    hot_days = df_forecast_logistic[df_forecast_logistic['predicted_sales'] == 1]
    hot_days = hot_days.sort_values('date')

    # สร้างข้อความละบรรทัด
    hot_text_lines = []
    for _, row in hot_days.iterrows():
        d = row['date']
        text_line = f"วันที่ {d.day:02d} เดือน {d.month:02d} : ขายดี !"
        hot_text_lines.append(text_line)

    final_text = "<br>".join(hot_text_lines) if hot_text_lines else "ไม่มีข้อมูลขายดี"

    # กล่องข้อความมุมล่างซ้าย
    fig_monthly.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.05,
        text=final_text,
        showarrow=False,
        align='left',
        font=dict(size=12, color="green"),
        bgcolor="rgba(255, 255, 255, 0.7)",
        bordercolor="green",
        borderwidth=1
    )

    fig_monthly.update_xaxes(title="เดือน (Jan–Dec)")
    fig_monthly.update_yaxes(title="ยอดขาย (บาท)")
    
    max_value = max(avg_sales_per_month['avg_sales'].max(), threshold)
    fig_monthly.update_yaxes(range=[0, max_value])
    
    fig_monthly.update_layout(height=400, margin=dict(l=20, r=20, t=60, b=20))

    monthly_graph_html = pio.to_html(fig_monthly, full_html=False)
    return Markup(monthly_graph_html)

#--------------------------------------------------------------------

# (6) Prophet Dashboard Endpoint 
@app.route('/dashboards/analytics')
def analytics_prophet():
    df = get_sales_data()
    df_prophet = preprocess_data_prophet(df)
    df_forecast = forecast_sales(df_prophet)

    # Generate plots and save HTML
    #save_plot_all_sales_vs_avg(df_to_calculate)
    #save_plot_7_day_forecast(df_forecast)

    # สร้างข้อมูล summary status
    summary = summary_status().get_json()

    # สร้าง weekly summary
    weekly = get_weekly_summary_data()
    
    # สร้าง top_trend
    trend_summary = google_trends_summary().get_json()

    # ส่งไปแสดงใน dashboard
    return render_template("dashboards/dashboard-analytics-prophet.html",
                       last_updated_date=df_prophet['ds'].max().strftime('%Y-%m-%d'),
                       weekly_cost=weekly["weekly_cost"], 
                       weekly_sales=weekly["weekly_sales"],
                       weekly_profit=weekly["weekly_profit"],
                       summary_status=summary,
                       top_trends=trend_summary["top_10_trends"])
#----------------------------------------------------------------------

# (6) Linear Regression Dashboard Endpoint 
@app.route('/dashboard-analytics-linear-regression/')
def analytics_linear():
    df = get_sales_data()
    df_lr, df_lr_plot = preprocess_data_linear(df)
    df_forecast_linear = forecast_sales_linear(df_lr)

    # Summary สำหรับการ์ด 4 กล่อง
    summary = summary_status_linear(df_forecast_linear)
    weekly = get_weekly_summary_data()
    trend_summary = google_trends_summary().get_json()

    return render_template("dashboards/dashboard-analytics-linear-regression.html",
                       last_updated_date=df_lr['date'].max().strftime('%Y-%m-%d'),
                       weekly_cost=weekly["weekly_cost"], 
                       weekly_sales=weekly["weekly_sales"],
                       weekly_profit=weekly["weekly_profit"],
                       summary_status_linear=summary,
                       top_trends=trend_summary["top_10_trends"])
#---------------------------------------------------------------------

# (6) Logistic Regression Dashboard Endpoint 
@app.route('/dashboard-analytics-logistic-regression/')
def analytics_logistic():
    df = get_sales_data()
    df_lg, df_lg_plot = preprocess_data_logistic(df)
    df_forecast_logistic = forecast_sales_logistic(df_lg)

    # สร้างข้อมูล summary status 
    summary = summary_status_logistic(df_forecast_logistic)

    # สร้าง weekly summary
    weekly = get_weekly_summary_data()

    # สร้าง top_trend
    trend_summary = google_trends_summary().get_json()

    # ส่งไปแสดงใน dashboard
    return render_template("dashboards/dashboard-analytics-logistic-regression.html",
                           last_updated_date=df_lg['date'].max().strftime('%Y-%m-%d'),
                           weekly_cost=weekly["weekly_cost"], 
                           weekly_sales=weekly["weekly_sales"],
                           weekly_profit=weekly["weekly_profit"],
                           summary_status_logistic=summary,
                           top_trends=trend_summary["top_10_trends"])
#---------------------------------------------------------------------

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

