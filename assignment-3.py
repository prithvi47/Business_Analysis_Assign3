# passenger_forecast_advanced.py
# Run: python passenger_forecast_advanced.py
# Then open http://127.0.0.1:5000

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
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
# Holt-Winters model
# -----------------------------
def fit_holt_winters(train, test):
    model = ExponentialSmoothing(
        train['passengers'],
        trend='add',
        seasonal='mul',
        seasonal_periods=12
    ).fit()
    forecast = model.forecast(len(test))
    return model, forecast

# -----------------------------
# SARIMA model (auto-order)
# -----------------------------
def fit_sarima(train, test):
    # Simplified: use fixed order (1,1,1)(1,1,1,12) – works well for AirPassengers
    model = ARIMA(train['passengers'], order=(1,1,1), seasonal_order=(1,1,1,12)).fit()
    forecast = model.forecast(len(test))
    return model, forecast

# -----------------------------
# Naive (last value) model
# -----------------------------
def fit_naive(train, test):
    last = train['passengers'].iloc[-1]
    forecast = pd.Series([last]*len(test), index=test.index)
    return None, forecast

# -----------------------------
# Evaluate model
# -----------------------------
def evaluate(test, forecast):
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test['passengers'] - forecast) / test['passengers'])) * 100
    return mae, rmse, mape

# -----------------------------
# Anomaly detection (residuals > 2 std)
# -----------------------------
def detect_anomalies(actual, forecast):
    residuals = actual['passengers'] - forecast
    threshold = 2 * residuals.std()
    anomalies = residuals[abs(residuals) > threshold]
    return anomalies

# -----------------------------
# Generate all plots (base64)
# -----------------------------
def generate_plots(df, hw_forecast, sarima_forecast, train_end_date, growth_scenario=0):
    plots = {}
    
    # 1. Comparison chart (Holt-Winters vs SARIMA vs actual)
    train = df.loc[:train_end_date]
    test = df.loc[train_end_date:].iloc[1:] if train_end_date != df.index[-1] else pd.DataFrame()
    
    fig1, ax1 = plt.subplots(figsize=(14,5))
    ax1.plot(train.index, train['passengers'], label='Training', color='#2E86AB')
    if len(test) > 0:
        ax1.plot(test.index, test['passengers'], label='Actual', color='#A23B72', marker='o')
        ax1.plot(test.index, hw_forecast, label='Holt-Winters', color='#F18F01', linestyle='--', marker='x')
        ax1.plot(test.index, sarima_forecast, label='SARIMA', color='#2CA02C', linestyle='-.', marker='s')
    ax1.set_title('Model Comparison: Holt-Winters vs SARIMA')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Passengers (thousands)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    plots['comparison'] = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig1)
    
    # 2. Decomposition (trend, seasonal, residual)
    decomp = seasonal_decompose(df['passengers'], model='multiplicative', period=12)
    fig2, axes = plt.subplots(3,1, figsize=(12,7), sharex=True)
    decomp.trend.plot(ax=axes[0], color='green', title='Trend')
    decomp.seasonal.plot(ax=axes[1], color='orange', title='Seasonal (12-month)')
    decomp.resid.plot(ax=axes[2], color='red', title='Residual')
    axes[2].set_xlabel('Date')
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    plots['decomposition'] = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig2)
    
    # 3. Seasonal bar chart
    monthly_avg = df.groupby(df.index.month)['passengers'].mean()
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    fig3, ax3 = plt.subplots(figsize=(10,5))
    colors = ['#e94560' if i in [6,7] else '#2E86AB' for i in range(12)]
    ax3.bar(months, monthly_avg.values, color=colors, edgecolor='black')
    ax3.axhline(y=df['passengers'].mean(), color='red', linestyle='--', label=f'Avg: {df["passengers"].mean():.0f}')
    ax3.set_title('Average Passengers by Month')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Passengers (thousands)')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
    plots['seasonal_bar'] = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig3)
    
    # 4. Forecast with scenario (if growth_scenario != 0)
    if growth_scenario != 0:
        # Apply growth multiplier to the forecast
        future_months = 12
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_months, freq='M')
        # Use Holt-Winters to forecast base
        hw_full = ExponentialSmoothing(df['passengers'], trend='add', seasonal='mul', seasonal_periods=12).fit()
        base_forecast = hw_full.forecast(future_months)
        # Apply scenario growth (multiplicative)
        scenario_forecast = base_forecast * (1 + growth_scenario/100)
        fig4, ax4 = plt.subplots(figsize=(12,5))
        ax4.plot(df.index, df['passengers'], label='Historical', color='blue')
        ax4.plot(future_dates, base_forecast, label='Base Forecast', color='gray', linestyle='--')
        ax4.plot(future_dates, scenario_forecast, label=f'{growth_scenario:+}% Growth Scenario', color='#F18F01', linestyle='-', linewidth=2)
        ax4.set_title('Scenario Analysis: What-If Growth')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Passengers (thousands)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        buf = io.BytesIO(); plt.savefig(buf, format='png'); buf.seek(0)
        plots['scenario'] = base64.b64encode(buf.getvalue()).decode()
        plt.close(fig4)
    
    return plots

# -----------------------------
# Financial / capacity calculations
# -----------------------------
def calculate_business_metrics(df, hw_forecast, avg_fare=150, seats_per_flight=180):
    """
    avg_fare: average ticket price ($)
    seats_per_flight: typical aircraft capacity
    """
    # Revenue
    historical_revenue = (df['passengers'] * avg_fare).sum()
    forecast_revenue = (hw_forecast * avg_fare).sum()
    # Required flights (assuming 80% load factor)
    load_factor = 0.8
    required_flights = hw_forecast / (seats_per_flight * load_factor)
    return {
        'historical_revenue': historical_revenue,
        'forecast_revenue': forecast_revenue,
        'required_flights_peak': required_flights.max(),
        'required_flights_avg': required_flights.mean()
    }

# -----------------------------
# Flask routes
# -----------------------------
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/forecast')
def api_forecast():
    df = load_data()
    train_end = request.args.get('train_end', default='1958-12-31')
    growth = float(request.args.get('growth', 0))
    train_end_date = pd.to_datetime(train_end)
    train = df.loc[:train_end_date]
    test = df.loc[train_end_date:].iloc[1:] if train_end_date != df.index[-1] else pd.DataFrame()
    
    # Fit models
    hw_model, hw_forecast = fit_holt_winters(train, test)
    sarima_model, sarima_forecast = fit_sarima(train, test)
    _, naive_forecast = fit_naive(train, test)
    
    # Evaluate
    if len(test) > 0:
        hw_mae, hw_rmse, hw_mape = evaluate(test, hw_forecast)
        sarima_mae, sarima_rmse, sarima_mape = evaluate(test, sarima_forecast)
        naive_mae, naive_rmse, naive_mape = evaluate(test, naive_forecast)
        best_model = "Holt-Winters" if hw_mape < sarima_mape else "SARIMA"
        best_mape = min(hw_mape, sarima_mape)
    else:
        hw_mape = sarima_mape = best_mape = 0
        best_model = "Holt-Winters"
    
    # Anomalies
    if len(test) > 0:
        anomalies = detect_anomalies(test, hw_forecast)
        anomaly_dates = anomalies.index.strftime('%Y-%m').tolist() if len(anomalies) > 0 else []
    else:
        anomaly_dates = []
    
    # Business metrics
    biz = calculate_business_metrics(train, hw_forecast if len(test) > 0 else hw_model.forecast(12))
    
    # Generate plots (including scenario if growth != 0)
    plots = generate_plots(df, hw_forecast if len(test) > 0 else pd.Series(), 
                           sarima_forecast if len(test) > 0 else pd.Series(), 
                           train_end_date, growth)
    
    # Seasonal peak info
    monthly_avg = df.groupby(df.index.month)['passengers'].mean()
    peak_month_num = monthly_avg.idxmax()
    peak_month_name = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][peak_month_num-1]
    seasonal_peak_pct = round((monthly_avg.max() / df['passengers'].mean() - 1) * 100, 1)
    
    return jsonify({
        'total_passengers': int(df['passengers'].sum()),
        'hw_mape': round(hw_mape, 1),
        'sarima_mape': round(sarima_mape, 1),
        'best_model': best_model,
        'best_mape': round(best_mape, 1),
        'peak_month': peak_month_name,
        'seasonal_peak_pct': seasonal_peak_pct,
        'anomaly_dates': anomaly_dates,
        'historical_revenue': f"${biz['historical_revenue']:,.0f}",
        'forecast_revenue': f"${biz['forecast_revenue']:,.0f}",
        'required_flights_peak': round(biz['required_flights_peak']),
        'required_flights_avg': round(biz['required_flights_avg']),
        'plots': plots
    })

@app.route('/api/download_forecast')
def download_forecast():
    df = load_data()
    # Use Holt-Winters to forecast next 12 months
    hw_full = ExponentialSmoothing(df['passengers'], trend='add', seasonal='mul', seasonal_periods=12).fit()
    future_months = 12
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=future_months, freq='M')
    forecast = hw_full.forecast(future_months)
    output = pd.DataFrame({'Date': future_dates, 'Forecasted_Passengers': forecast})
    output.to_csv('forecast.csv', index=False)
    return send_file('forecast.csv', as_attachment=True, download_name='airline_forecast.csv')

# -----------------------------
# HTML Template (Tailwind + extra features)
# -----------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Advanced Airline Passenger Forecasting | Business Intelligence</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    <style>
        body { font-family: 'Inter', sans-serif; background: #F1F5F9; }
        .card { background: white; border-radius: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); transition: all 0.2s; }
        .card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
        .gradient-text { background: linear-gradient(135deg, #3B82F6, #8B5CF6); -webkit-background-clip: text; background-clip: text; color: transparent; }
        .loading { position: fixed; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,0.5); backdrop-filter: blur(4px); display: flex; justify-content: center; align-items: center; z-index: 1000; }
        .spinner { border: 3px solid rgba(255,255,255,0.3); border-top-color: white; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        input, select { border-radius: 0.75rem; border: 1px solid #cbd5e1; padding: 0.5rem 1rem; }
        button { transition: all 0.2s; }
        button:hover { transform: translateY(-1px); }
    </style>
</head>
<body>
    <div id="loading" class="loading hidden"><div class="spinner"></div></div>

    <div class="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-4xl font-extrabold tracking-tight"><span class="gradient-text">✈️ Airline Passenger Forecasting</span></h1>
            <p class="text-gray-600 mt-2">Holt‑Winters vs SARIMA | Capacity Planning | Revenue Impact | Scenario Analysis</p>
        </div>

        <!-- Controls -->
        <div class="card p-5 mb-8 bg-white/80 backdrop-blur-sm">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
                <div>
                    <label class="block text-sm font-medium text-gray-700">Training end date</label>
                    <input type="date" id="trainEnd" value="1958-12-31" class="w-full mt-1">
                </div>
                <div>
                    <label class="block text-sm font-medium text-gray-700">Growth scenario (%)</label>
                    <input type="number" id="growthScenario" value="0" step="5" class="w-full mt-1">
                    <p class="text-xs text-gray-500 mt-1">What-if: +5% = 5% higher future demand</p>
                </div>
                <div>
                    <button id="refreshBtn" class="bg-gradient-to-r from-blue-600 to-purple-600 text-white px-5 py-2 rounded-xl font-medium w-full"><i class="fas fa-sync-alt mr-2"></i>Update Forecast</button>
                </div>
            </div>
            <div class="mt-4 flex gap-3">
                <button id="downloadBtn" class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-xl text-sm"><i class="fas fa-download mr-2"></i>Download Forecast CSV</button>
            </div>
        </div>

        <!-- KPI Grid -->
        <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <div class="card p-5"><div class="text-gray-500 text-sm">Total Passengers (historical)</div><div class="text-2xl font-bold" id="totalPass">--</div></div>
            <div class="card p-5"><div class="text-gray-500 text-sm">Best Model MAPE</div><div class="text-2xl font-bold" id="bestMape">--</div><div class="text-green-600 text-sm">↓ lower is better</div></div>
            <div class="card p-5"><div class="text-gray-500 text-sm">Peak Month</div><div class="text-2xl font-bold" id="peakMonth">--</div><div class="text-orange-500 text-sm">↑ +<span id="peakPct">--</span>% above avg</div></div>
            <div class="card p-5"><div class="text-gray-500 text-sm">Required Flights (peak month)</div><div class="text-2xl font-bold" id="reqFlights">--</div><div class="text-blue-600 text-sm">based on 180 seats, 80% load</div></div>
        </div>

        <!-- Revenue & Capacity Cards -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
            <div class="card p-5"><div class="text-gray-500 text-sm">Historical Revenue</div><div class="text-2xl font-bold" id="histRevenue">--</div></div>
            <div class="card p-5"><div class="text-gray-500 text-sm">Forecast Revenue (next 12 months)</div><div class="text-2xl font-bold" id="foreRevenue">--</div></div>
        </div>

        <!-- Comparison Chart -->
        <div class="card p-5 mb-8">
            <h2 class="text-xl font-bold mb-4"><i class="fas fa-chart-line text-blue-500 mr-2"></i>Model Comparison: Holt-Winters vs SARIMA</h2>
            <img id="comparisonPlot" class="w-full rounded-lg" alt="Comparison Chart">
        </div>

        <!-- Decomposition & Seasonality -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
            <div class="card p-5"><h2 class="text-xl font-bold mb-4"><i class="fas fa-chart-pie text-purple-500 mr-2"></i>Trend & Seasonality</h2><img id="decompPlot" class="w-full rounded-lg" alt="Decomposition"></div>
            <div class="card p-5"><h2 class="text-xl font-bold mb-4"><i class="fas fa-calendar-alt text-orange-500 mr-2"></i>Monthly Seasonal Pattern</h2><img id="seasonalPlot" class="w-full rounded-lg" alt="Seasonal Bar"></div>
        </div>

        <!-- Scenario Plot (conditional) -->
        <div id="scenarioContainer" class="card p-5 mb-8 hidden">
            <h2 class="text-xl font-bold mb-4"><i class="fas fa-chart-line text-yellow-500 mr-2"></i>Scenario Analysis</h2>
            <img id="scenarioPlot" class="w-full rounded-lg" alt="Scenario">
        </div>

        <!-- Anomaly Detection -->
        <div class="card p-5 mb-8">
            <h2 class="text-xl font-bold mb-2"><i class="fas fa-exclamation-triangle text-red-500 mr-2"></i>Anomaly Detection</h2>
            <p id="anomalyText" class="text-gray-600">Loading...</p>
        </div>

        <!-- Operational Insights -->
        <div class="card p-5 bg-gradient-to-r from-indigo-50 to-purple-50">
            <h2 class="text-xl font-bold mb-4"><i class="fas fa-lightbulb text-yellow-500 mr-2"></i>Operational & Strategic Insights</h2>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4" id="insights"></div>
        </div>

        <div class="text-center text-gray-500 text-sm mt-8 border-t pt-6">Data: AirPassengers (1949-1960) | Holt-Winters (add trend, mul season) | SARIMA (1,1,1)(1,1,1,12) | Average fare $150</div>
    </div>

    <script>
        async function loadData() {
            showLoading();
            const trainEnd = document.getElementById('trainEnd').value;
            const growth = document.getElementById('growthScenario').value;
            const res = await fetch(`/api/forecast?train_end=${trainEnd}&growth=${growth}`);
            const data = await res.json();
            
            // KPIs
            document.getElementById('totalPass').innerText = data.total_passengers.toLocaleString();
            document.getElementById('bestMape').innerText = data.best_mape + '%';
            document.getElementById('peakMonth').innerText = data.peak_month;
            document.getElementById('peakPct').innerText = data.seasonal_peak_pct;
            document.getElementById('reqFlights').innerText = data.required_flights_peak;
            document.getElementById('histRevenue').innerText = data.historical_revenue;
            document.getElementById('foreRevenue').innerText = data.forecast_revenue;
            
            // Plots
            document.getElementById('comparisonPlot').src = 'data:image/png;base64,' + data.plots.comparison;
            document.getElementById('decompPlot').src = 'data:image/png;base64,' + data.plots.decomposition;
            document.getElementById('seasonalPlot').src = 'data:image/png;base64,' + data.plots.seasonal_bar;
            
            // Scenario plot
            if (data.plots.scenario) {
                document.getElementById('scenarioContainer').classList.remove('hidden');
                document.getElementById('scenarioPlot').src = 'data:image/png;base64,' + data.plots.scenario;
            } else {
                document.getElementById('scenarioContainer').classList.add('hidden');
            }
            
            // Anomalies
            const anomalyHtml = data.anomaly_dates.length ? 
                `<i class="fas fa-chart-line"></i> Unusual months detected: ${data.anomaly_dates.join(', ')}. Review operations for those periods.` :
                `<i class="fas fa-check-circle text-green-500"></i> No significant anomalies detected. Forecast residuals are stable.`;
            document.getElementById('anomalyText').innerHTML = anomalyHtml;
            
            // Insights
            const insightsHtml = `
                <div class="flex gap-3"><i class="fas fa-users text-blue-500 text-xl"></i><div><strong>Staffing & Capacity</strong><br>Peak month requires ~${data.required_flights_peak} flights (based on 180 seats, 80% load). Increase crew by 30% in ${data.peak_month}.</div></div>
                <div class="flex gap-3"><i class="fas fa-dollar-sign text-green-500 text-xl"></i><div><strong>Revenue Management</strong><br>Forecast revenue next 12 months: ${data.forecast_revenue}. Implement dynamic pricing: +20% in peak months, -15% in troughs.</div></div>
                <div class="flex gap-3"><i class="fas fa-chart-line text-purple-500 text-xl"></i><div><strong>Model Selection</strong><br>Best model: ${data.best_model} (MAPE = ${data.best_mape}%). Use hybrid (SARIMA + HW) for improved accuracy.</div></div>
                <div class="flex gap-3"><i class="fas fa-calendar-week text-orange-500 text-xl"></i><div><strong>Maintenance Scheduling</strong><br>Schedule heavy maintenance in February and November (lowest demand).</div></div>
            `;
            document.getElementById('insights').innerHTML = insightsHtml;
            hideLoading();
        }
        
        function showLoading() { document.getElementById('loading').classList.remove('hidden'); }
        function hideLoading() { document.getElementById('loading').classList.add('hidden'); }
        
        document.getElementById('refreshBtn').addEventListener('click', () => loadData());
        document.getElementById('downloadBtn').addEventListener('click', async () => {
            window.location.href = '/api/download_forecast';
        });
        
        loadData();
        setInterval(loadData, 300000); // auto-refresh every 5 min
    </script>
</body>
</html>
"""

# -----------------------------
# Main
# -----------------------------
if __name__ == '__main__':
    print("Starting Advanced Passenger Traffic Forecasting Web App...")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(debug=False, host='127.0.0.1', port=5000)
