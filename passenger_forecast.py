# passenger_forecast_fixed.py
# Fully working with proper dimension handling

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
import io
import base64
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template_string, jsonify, request, send_file

app = Flask(__name__)

# -----------------------------
# Data loading
# -----------------------------
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
    df = pd.read_csv(url, parse_dates=['Month'], index_col='Month')
    df.columns = ['passengers']
    return df

# -----------------------------
# Models
# -----------------------------
def fit_holt_winters(train, test):
    model = ExponentialSmoothing(train['passengers'], trend='add', seasonal='mul', seasonal_periods=12).fit()
    forecast = model.forecast(len(test))
    return model, forecast

def fit_sarima(train, test):
    model = ARIMA(train['passengers'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
    forecast = model.forecast(len(test))
    return model, forecast

def fit_naive(train, test):
    last = train['passengers'].iloc[-1]
    forecast = pd.Series([last]*len(test), index=test.index)
    return None, forecast

def evaluate(test, forecast):
    # Ensure same length
    min_len = min(len(test), len(forecast))
    test_aligned = test.iloc[:min_len]
    forecast_aligned = forecast.iloc[:min_len]
    mae = mean_absolute_error(test_aligned, forecast_aligned)
    rmse = np.sqrt(mean_squared_error(test_aligned, forecast_aligned))
    mape = np.mean(np.abs((test_aligned['passengers'] - forecast_aligned) / test_aligned['passengers'])) * 100
    return mae, rmse, mape

# -----------------------------
# Advanced plots (base64) – with length safety
# -----------------------------
def generate_all_plots(df, train, test, hw_forecast, sarima_forecast, anomalies, growth_scenario=0):
    plots = {}
    
    # Ensure all forecast series align with test index
    common_index = test.index
    hw_forecast = hw_forecast.reindex(common_index)
    sarima_forecast = sarima_forecast.reindex(common_index)
    
    # 1. Main forecast comparison
    fig1, ax1 = plt.subplots(figsize=(14,5))
    ax1.plot(train.index, train['passengers'], label='Training (1949-1958)', color='#2E86AB', linewidth=2)
    ax1.plot(test.index, test['passengers'], label='Actual (1959-1960)', color='#A23B72', marker='o', linewidth=2)
    ax1.plot(test.index, hw_forecast, label='Holt-Winters Forecast', color='#F18F01', linestyle='--', linewidth=2, marker='x')
    ax1.plot(test.index, sarima_forecast, label='SARIMA Forecast', color='#2CA02C', linestyle='-.', linewidth=2, marker='s')
    # Confidence band
    resid = test['passengers'] - hw_forecast
    if len(resid) > 1:
        std_resid = resid.std()
        ax1.fill_between(test.index, hw_forecast - std_resid, hw_forecast + std_resid, alpha=0.2, color='#F18F01', label='±1σ')
    ax1.set_title('Model Comparison: Holt-Winters vs SARIMA', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Passengers (thousands)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    plots['comparison'] = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig1)
    
    # 2. Decomposition
    decomp = seasonal_decompose(df['passengers'], model='multiplicative', period=12)
    fig2, axes = plt.subplots(3,1, figsize=(12,8), sharex=True)
    decomp.trend.plot(ax=axes[0], color='green', linewidth=2, title='Trend Component')
    decomp.seasonal.plot(ax=axes[1], color='orange', linewidth=2, title='Seasonal Component (12-month)')
    decomp.resid.plot(ax=axes[2], color='red', linewidth=1, title='Residual (Irregular)')
    axes[2].set_xlabel('Date')
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    plots['decomposition'] = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig2)
    
    # 3. Monthly seasonal bar chart
    monthly_avg = df.groupby(df.index.month)['passengers'].mean()
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig3, ax3 = plt.subplots(figsize=(10,5))
    colors = ['#e94560' if i in [6,7] else '#2E86AB' for i in range(12)]
    bars = ax3.bar(months, monthly_avg.values, color=colors, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=df['passengers'].mean(), color='red', linestyle='--', linewidth=2, label=f'Annual Avg: {df["passengers"].mean():.0f}')
    ax3.set_title('Average Passengers by Month (Seasonal Pattern)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Passengers (thousands)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, monthly_avg.values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, f'{val:.0f}', ha='center', fontsize=9)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    plots['seasonal'] = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig3)
    
    # 4. Anomaly detection plot (only if test set not empty and residuals exist)
    if len(test) > 0 and len(hw_forecast) == len(test):
        residuals = test['passengers'] - hw_forecast
        if len(residuals) > 1:
            fig4, ax4 = plt.subplots(figsize=(12,4))
            ax4.plot(test.index, residuals, color='gray', marker='o', linestyle='-', linewidth=1, markersize=4, label='Residuals')
            ax4.axhline(y=0, color='black', linestyle='-')
            std_res = residuals.std()
            ax4.axhline(y=2*std_res, color='red', linestyle='--', label='+2 Std Dev')
            ax4.axhline(y=-2*std_res, color='red', linestyle='--', label='-2 Std Dev')
            # Highlight anomalies
            anomaly_mask = abs(residuals) > 2*std_res
            anomaly_points = residuals[anomaly_mask]
            ax4.scatter(anomaly_points.index, anomaly_points.values, color='red', s=100, zorder=5, label='Anomalies')
            ax4.set_title('Anomaly Detection (Residuals > 2σ)', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Date')
            ax4.set_ylabel('Forecast Error (thousands)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            plt.tight_layout()
            buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
            plots['anomalies'] = base64.b64encode(buf.getvalue()).decode()
            plt.close(fig4)
    
    # 5. Capacity planning chart
    seats_per_flight = 180
    load_factor = 0.8
    required_flights = hw_forecast / (seats_per_flight * load_factor)
    fig5, ax5 = plt.subplots(figsize=(10,4))
    ax5.bar(test.index, required_flights, color='#3B82F6', edgecolor='black', alpha=0.7)
    ax5.axhline(y=required_flights.mean(), color='red', linestyle='--', label=f'Avg flights: {required_flights.mean():.0f}')
    ax5.set_title('Required Flights per Month (180 seats, 80% load factor)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Date')
    ax5.set_ylabel('Number of flights')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    plots['capacity'] = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig5)
    
    # 6. Scenario analysis (if growth_scenario != 0)
    if growth_scenario != 0:
        future_months = 12
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_months, freq='M')
        hw_full = ExponentialSmoothing(df['passengers'], trend='add', seasonal='mul', seasonal_periods=12).fit()
        base_forecast = hw_full.forecast(future_months)
        scenario_forecast = base_forecast * (1 + growth_scenario/100)
        fig6, ax6 = plt.subplots(figsize=(12,5))
        ax6.plot(df.index, df['passengers'], label='Historical', color='blue', linewidth=2)
        ax6.plot(future_dates, base_forecast, label='Base Forecast', color='gray', linestyle='--', linewidth=2)
        ax6.plot(future_dates, scenario_forecast, label=f'{growth_scenario:+}% Growth Scenario', color='#F18F01', linewidth=2.5)
        ax6.fill_between(future_dates, scenario_forecast*0.9, scenario_forecast*1.1, alpha=0.2, color='#F18F01')
        ax6.set_title('Scenario Analysis: What-If Growth', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Date')
        ax6.set_ylabel('Passengers (thousands)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
        plots['scenario'] = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig6)
    
    return plots

# -----------------------------
# Business metrics
# -----------------------------
def business_metrics(train, hw_forecast, avg_fare=150):
    hist_rev = (train['passengers'] * avg_fare).sum()
    forecast_rev = (hw_forecast * avg_fare).sum()
    return hist_rev, forecast_rev

# -----------------------------
# Flask routes
# -----------------------------
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/forecast')
def api_forecast():
    train_end = request.args.get('train_end', default='1958-12-31')
    growth = float(request.args.get('growth', 0))
    train_end_date = pd.to_datetime(train_end)
    df = load_data()
    train = df.loc[:train_end_date]
    # Ensure test includes all dates after train_end (excluding the train_end itself if it's in train)
    test = df.loc[train_end_date:].iloc[1:] if train_end_date != df.index[-1] else pd.DataFrame()
    
    # If test is empty (e.g., user selected end date too late), fallback to default split
    if len(test) == 0:
        train = df.iloc[:int(0.8*len(df))]
        test = df.iloc[int(0.8*len(df)):]
    
    # Fit models
    _, hw_forecast = fit_holt_winters(train, test)
    _, sarima_forecast = fit_sarima(train, test)
    _, naive_forecast = fit_naive(train, test)
    
    # Evaluate
    hw_mae, hw_rmse, hw_mape = evaluate(test, hw_forecast)
    sarima_mae, sarima_rmse, sarima_mape = evaluate(test, sarima_forecast)
    
    best_model = "Holt-Winters" if hw_mape < sarima_mape else "SARIMA"
    best_mape = min(hw_mape, sarima_mape)
    
    # Anomalies (Holt-Winters residuals)
    residuals = test['passengers'] - hw_forecast
    if len(residuals) > 1:
        anomalies = residuals[abs(residuals) > 2*residuals.std()]
        anomaly_dates = anomalies.index.strftime('%Y-%m').tolist() if len(anomalies) > 0 else []
    else:
        anomaly_dates = []
    
    # Business metrics
    hist_rev, forecast_rev = business_metrics(train, hw_forecast)
    
    # Seasonal info
    monthly_avg = df.groupby(df.index.month)['passengers'].mean()
    peak_month_num = monthly_avg.idxmax()
    peak_month_name = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][peak_month_num-1]
    seasonal_peak_pct = round((monthly_avg.max() / df['passengers'].mean() - 1) * 100, 1)
    
    # Generate plots
    plots = generate_all_plots(df, train, test, hw_forecast, sarima_forecast, anomalies, growth)
    
    # Capacity planning
    seats_per_flight = 180
    load_factor = 0.8
    required_flights_peak = (hw_forecast.max() / (seats_per_flight * load_factor))
    required_flights_avg = (hw_forecast.mean() / (seats_per_flight * load_factor))
    
    return jsonify({
        'total_passengers': int(df['passengers'].sum()),
        'hw_mape': round(hw_mape, 1),
        'sarima_mape': round(sarima_mape, 1),
        'best_model': best_model,
        'best_mape': round(best_mape, 1),
        'peak_month': peak_month_name,
        'seasonal_peak_pct': seasonal_peak_pct,
        'anomaly_dates': anomaly_dates,
        'historical_revenue': f"${hist_rev:,.0f}",
        'forecast_revenue': f"${forecast_rev:,.0f}",
        'required_flights_peak': round(required_flights_peak),
        'required_flights_avg': round(required_flights_avg),
        'plots': plots
    })

@app.route('/api/download_forecast')
def download_forecast():
    df = load_data()
    hw_full = ExponentialSmoothing(df['passengers'], trend='add', seasonal='mul', seasonal_periods=12).fit()
    future_months = 12
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_months, freq='M')
    forecast = hw_full.forecast(future_months)
    output = pd.DataFrame({'Date': future_dates, 'Forecasted_Passengers': forecast.round(0).astype(int)})
    output.to_csv('forecast.csv', index=False)
    return send_file('forecast.csv', as_attachment=True, download_name='airline_forecast.csv')

# -----------------------------
# HTML Template (same as before)
# -----------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>✈️ Airline Passenger Intelligence | Forecast Hub</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        body { font-family: 'Inter', sans-serif; background: #F1F5F9; }
        .card { background: white; border-radius: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); transition: all 0.2s; }
        .card:hover { transform: translateY(-4px); box-shadow: 0 20px 25px -5px rgba(0,0,0,0.1); }
        .gradient-text { background: linear-gradient(135deg, #3B82F6, #8B5CF6); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .kpi-card { background: linear-gradient(135deg, #1E293B, #0F172A); color: white; }
        .tooltip { position: relative; display: inline-block; cursor: help; border-bottom: 1px dotted #999; }
        .tooltip .tooltiptext { visibility: hidden; width: 200px; background-color: #333; color: #fff; text-align: center; border-radius: 6px; padding: 5px; position: absolute; z-index: 1; bottom: 125%; left: 50%; margin-left: -100px; opacity: 0; transition: opacity 0.3s; font-size: 0.75rem; }
        .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
        .loading { position: fixed; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,0.6); backdrop-filter: blur(4px); display: flex; justify-content: center; align-items: center; z-index: 1000; }
        .spinner { border: 4px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        input, select { border-radius: 0.75rem; border: 1px solid #cbd5e1; padding: 0.5rem 1rem; }
        button { transition: all 0.2s; }
        button:hover { transform: translateY(-2px); }
        .insight-card { background: linear-gradient(135deg, #EFF6FF, #F3E8FF); border-left: 4px solid #3B82F6; }
    </style>
</head>
<body>
    <div id="loading" class="loading hidden"><div class="spinner"></div></div>
    <div class="max-w-7xl mx-auto px-4 py-8">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl font-extrabold tracking-tight"><span class="gradient-text">✈️ Airline Passenger Intelligence</span></h1>
            <p class="text-gray-600 mt-2">Holt‑Winters · SARIMA · Capacity Planning · Anomaly Detection</p>
        </div>
        
        <!-- Controls -->
        <div class="card p-5 mb-8 bg-white/80 backdrop-blur-sm">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
                <div>
                    <label class="block text-sm font-medium text-gray-700">Training end date <span class="tooltip">ⓘ<span class="tooltiptext">Data before this date used for training</span></span></label>
                    <input type="date" id="trainEnd" value="1958-12-31" class="w-full mt-1">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700">Growth scenario (%)</label>
                    <input type="range" id="growthSlider" min="-20" max="30" value="0" step="5" class="w-full">
                    <p class="text-xs text-gray-500 mt-1">Current: <span id="growthValue">0</span>% (what-if demand change)</p>
                </div>
                <div>
                    <button id="refreshBtn" class="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-5 py-2 rounded-xl font-medium w-full"><i class="fas fa-sync-alt mr-2"></i>Update Forecast</button>
                </div>
            </div>
            <div class="mt-4">
                <button id="downloadBtn" class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-xl text-sm"><i class="fas fa-download mr-2"></i>Download 12-Month Forecast CSV</button>
            </div>
        </div>
        
        <!-- KPI Row -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="card p-5"><div class="text-gray-500 text-sm">Total Passengers (historical)</div><div class="text-2xl font-bold" id="totalPass">--</div></div>
            <div class="card p-5"><div class="text-gray-500 text-sm">Best Model MAPE</div><div class="text-2xl font-bold" id="bestMape">--</div><div class="text-green-600 text-sm">↓ lower is better</div></div>
            <div class="card p-5"><div class="text-gray-500 text-sm">Peak Month</div><div class="text-2xl font-bold" id="peakMonth">--</div><div class="text-orange-500 text-sm">↑ +<span id="peakPct">--</span>% above avg</div></div>
            <div class="card p-5"><div class="text-gray-500 text-sm">Required Flights (peak)</div><div class="text-2xl font-bold" id="reqFlights">--</div><div class="text-blue-600 text-sm">based on 180 seats, 80% load</div></div>
        </div>
        
        <!-- Revenue & Accuracy -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div class="card p-5"><div class="text-gray-500 text-sm">Historical Revenue (1949-1958)</div><div class="text-2xl font-bold" id="histRevenue">--</div><div class="text-gray-500 text-sm">at $150 avg fare</div></div>
            <div class="card p-5"><div class="text-gray-500 text-sm">Forecast Revenue (next 12 months)</div><div class="text-2xl font-bold" id="foreRevenue">--</div><div class="text-green-600 text-sm"><i class="fas fa-chart-line"></i> projected</div></div>
        </div>
        
        <!-- Main Chart: Model Comparison -->
        <div class="card p-5 mb-8">
            <h2 class="text-xl font-bold mb-4"><i class="fas fa-chart-line text-blue-500 mr-2"></i>Model Comparison: Holt-Winters vs SARIMA</h2>
            <img id="comparisonPlot" class="w-full rounded-lg" alt="Comparison Chart">
        </div>
        
        <!-- Decomposition & Seasonality -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <div class="card p-5"><h2 class="text-xl font-bold mb-4"><i class="fas fa-chart-pie text-purple-500 mr-2"></i>Trend & Seasonality Decomposition</h2><img id="decompPlot" class="w-full rounded-lg" alt="Decomposition"></div>
            <div class="card p-5"><h2 class="text-xl font-bold mb-4"><i class="fas fa-calendar-alt text-orange-500 mr-2"></i>Monthly Seasonal Pattern</h2><img id="seasonalPlot" class="w-full rounded-lg" alt="Seasonal Bar"></div>
        </div>
        
        <!-- Capacity Planning & Anomalies -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <div class="card p-5"><h2 class="text-xl font-bold mb-4"><i class="fas fa-plane text-blue-500 mr-2"></i>Capacity Planning (Flights Required)</h2><img id="capacityPlot" class="w-full rounded-lg" alt="Capacity"></div>
            <div class="card p-5"><h2 class="text-xl font-bold mb-4"><i class="fas fa-exclamation-triangle text-red-500 mr-2"></i>Anomaly Detection</h2><div id="anomalyContainer"><img id="anomalyPlot" class="w-full rounded-lg" alt="Anomalies"><p id="anomalyText" class="text-sm text-gray-600 mt-2"></p></div></div>
        </div>
        
        <!-- Scenario Analysis (conditional) -->
        <div id="scenarioContainer" class="card p-5 mb-8 hidden">
            <h2 class="text-xl font-bold mb-4"><i class="fas fa-chart-line text-yellow-500 mr-2"></i>Scenario Analysis</h2>
            <img id="scenarioPlot" class="w-full rounded-lg" alt="Scenario">
        </div>
        
        <!-- Operational Insights -->
        <div class="card p-5 mb-8 insight-card">
            <h2 class="text-xl font-bold mb-4"><i class="fas fa-lightbulb text-yellow-500 mr-2"></i>Operational & Strategic Insights</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4" id="insights"></div>
        </div>
        
        <div class="text-center text-gray-500 text-sm mt-8 border-t pt-6">
            <i class="fas fa-chart-line"></i> Data: AirPassengers (1949-1960) | Holt-Winters (add trend, mul season) | SARIMA (1,1,1)(1,1,1,12) | Avg fare $150
        </div>
    </div>
    
    <script>
        async function loadData() {
            showLoading();
            const trainEnd = document.getElementById('trainEnd').value;
            const growth = document.getElementById('growthSlider').value;
            document.getElementById('growthValue').innerText = growth;
            const res = await fetch(`/api/forecast?train_end=${trainEnd}&growth=${growth}`);
            const data = await res.json();
            
            // KPIs
            document.getElementById('totalPass').innerText = data.total_passengers.toLocaleString();
            document.getElementById('bestMape').innerHTML = data.best_mape + '%';
            document.getElementById('peakMonth').innerText = data.peak_month;
            document.getElementById('peakPct').innerText = data.seasonal_peak_pct;
            document.getElementById('reqFlights').innerText = data.required_flights_peak;
            document.getElementById('histRevenue').innerText = data.historical_revenue;
            document.getElementById('foreRevenue').innerText = data.forecast_revenue;
            
            // Plots
            document.getElementById('comparisonPlot').src = 'data:image/png;base64,' + data.plots.comparison;
            document.getElementById('decompPlot').src = 'data:image/png;base64,' + data.plots.decomposition;
            document.getElementById('seasonalPlot').src = 'data:image/png;base64,' + data.plots.seasonal;
            document.getElementById('capacityPlot').src = 'data:image/png;base64,' + data.plots.capacity;
            
            // Anomalies
            if (data.plots.anomalies) {
                document.getElementById('anomalyPlot').src = 'data:image/png;base64,' + data.plots.anomalies;
                document.getElementById('anomalyText').innerHTML = data.anomaly_dates.length ? 
                    `<i class="fas fa-chart-line"></i> Unusual months detected: ${data.anomaly_dates.join(', ')}. Review operations.` :
                    `<i class="fas fa-check-circle text-green-500"></i> No significant anomalies detected. Forecast residuals are stable.`;
            } else {
                document.getElementById('anomalyContainer').innerHTML = '<p class="text-gray-500">Anomaly plot not available for this date range.</p>';
            }
            
            // Scenario
            if (data.plots.scenario) {
                document.getElementById('scenarioContainer').classList.remove('hidden');
                document.getElementById('scenarioPlot').src = 'data:image/png;base64,' + data.plots.scenario;
            } else {
                document.getElementById('scenarioContainer').classList.add('hidden');
            }
            
            // Insights
            document.getElementById('insights').innerHTML = `
                <div class="flex gap-3"><i class="fas fa-users text-blue-500 text-xl"></i><div><strong>Peak Season Planning</strong><br>July–August demand is ${data.seasonal_peak_pct}% above average. Increase staffing & flights by 25‑30%.</div></div>
                <div class="flex gap-3"><i class="fas fa-tools text-gray-600 text-xl"></i><div><strong>Maintenance Scheduling</strong><br>Lowest demand in February/November – schedule heavy maintenance then.</div></div>
                <div class="flex gap-3"><i class="fas fa-dollar-sign text-green-500 text-xl"></i><div><strong>Dynamic Pricing</strong><br>Implement premium fares in summer (+20%), promotions in trough months (-15%).</div></div>
                <div class="flex gap-3"><i class="fas fa-robot text-purple-500 text-xl"></i><div><strong>Hybrid Forecasting</strong><br>Improve MAPE by 10‑20% using SARIMA, Prophet, or XGBoost on residuals.</div></div>
                <div class="flex gap-3"><i class="fas fa-chart-line text-orange-500 text-xl"></i><div><strong>Capacity Utilisation</strong><br>Peak month requires ~${data.required_flights_peak} flights (based on 180 seats, 80% load).</div></div>
                <div class="flex gap-3"><i class="fas fa-chart-pie text-indigo-500 text-xl"></i><div><strong>Revenue Opportunity</strong><br>Forecast revenue next 12 months: ${data.forecast_revenue}. Upside from dynamic pricing: +5-10%.</div></div>
            `;
            hideLoading();
        }
        
        function showLoading() { document.getElementById('loading').classList.remove('hidden'); }
        function hideLoading() { document.getElementById('loading').classList.add('hidden'); }
        
        document.getElementById('refreshBtn').addEventListener('click', () => loadData());
        document.getElementById('growthSlider').addEventListener('input', function() {
            document.getElementById('growthValue').innerText = this.value;
        });
        document.getElementById('downloadBtn').addEventListener('click', async () => {
            window.location.href = '/api/download_forecast';
        });
        
        loadData();
        setInterval(loadData, 300000);
    </script>
</body>
</html>
"""

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    print("Starting Fixed Passenger Traffic Forecasting Web App...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, host='127.0.0.1', port=5000)
