# app_fixed.py - Fixed version with no 403 error
# Save this file and run: python app_fixed.py

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid threading issues
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import SimpleExpSmoothing, Holt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, render_template, jsonify, send_from_directory
import io
import base64
import os
import webbrowser
import threading
import time

# Create Flask app with explicit static folder
app = Flask(__name__, static_folder='static', template_folder='templates')

# Global variables
sales_data = None
forecast_results = None
current_dataset_source = "synthetic"

# -----------------------------
# Generate Synthetic Data (No Kaggle dependency for reliability)
# -----------------------------
def generate_sales_data():
    """Generate realistic synthetic retail sales data"""
    np.random.seed(42)
    dates = pd.date_range(start='2021-01-01', periods=36, freq='M')
    months = np.arange(1, 37)
    
    # Base trend: growing business
    trend = 100 + 6 * (months / 12)
    
    # Seasonal spikes
    seasonal = np.zeros(36)
    for i, date in enumerate(dates):
        if date.month == 12:  # Christmas peak
            seasonal[i] = np.random.uniform(55, 75)
        elif date.month == 7:  # Summer sale
            seasonal[i] = np.random.uniform(15, 25)
        elif date.month == 1:  # Post-holiday dip
            seasonal[i] = np.random.uniform(-10, -5)
    
    # Random noise
    noise = np.random.normal(0, 7, 36)
    
    # Final sales
    sales = trend + seasonal + noise
    sales = np.maximum(sales, 15)
    
    df = pd.DataFrame({'Sales': sales}, index=dates)
    
    # Add metadata
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Year'] = df.index.year
    
    return df

# -----------------------------
# Forecasting Models
# -----------------------------
def run_forecasting(df):
    """Run all forecasting models"""
    # Split data (80% train, 20% test)
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    predictions = {}
    
    # Model 1: Naive Forecast
    naive_pred = [train['Sales'].iloc[-1]] * len(test)
    predictions['Naive (Baseline)'] = pd.Series(naive_pred, index=test.index)
    
    # Model 2: Moving Average (window=3)
    ma_forecast = train['Sales'].rolling(window=3).mean().iloc[-1]
    predictions['Moving Average (w=3)'] = pd.Series([ma_forecast] * len(test), index=test.index)
    
    # Model 3: Simple Exponential Smoothing
    try:
        ses_model = SimpleExpSmoothing(train['Sales']).fit(optimized=True)
        predictions['Simple Exponential Smoothing'] = pd.Series(ses_model.forecast(len(test)), index=test.index)
    except:
        predictions['Simple Exponential Smoothing'] = predictions['Moving Average (w=3)']
    
    # Model 4: Holt's Linear Trend
    try:
        holt_model = Holt(train['Sales']).fit(optimized=True)
        predictions["Holt's Linear Trend"] = pd.Series(holt_model.forecast(len(test)), index=test.index)
    except:
        predictions["Holt's Linear Trend"] = predictions['Simple Exponential Smoothing']
    
    # Model 5: Holt-Winters Additive (Best for retail)
    try:
        hw_model = ExponentialSmoothing(
            train['Sales'],
            trend='add',
            seasonal='add',
            seasonal_periods=12
        ).fit(optimized=True)
        predictions['Holt-Winters Additive'] = pd.Series(hw_model.forecast(len(test)), index=test.index)
    except:
        predictions['Holt-Winters Additive'] = predictions["Holt's Linear Trend"]
    
    # Model 6: Holt-Winters Damped
    try:
        hw_damped = ExponentialSmoothing(
            train['Sales'],
            trend='add',
            damped_trend=True,
            seasonal='add',
            seasonal_periods=12
        ).fit(optimized=True)
        predictions['Holt-Winters Damped'] = pd.Series(hw_damped.forecast(len(test)), index=test.index)
    except:
        predictions['Holt-Winters Damped'] = predictions['Holt-Winters Additive']
    
    # Calculate metrics
    results = []
    for name, pred in predictions.items():
        if len(test) > 0 and len(pred) > 0:
            mae = mean_absolute_error(test['Sales'], pred)
            rmse = np.sqrt(mean_squared_error(test['Sales'], pred))
            mape = np.mean(np.abs((test['Sales'].values - pred.values) / test['Sales'].values)) * 100
            
            results.append({
                'model': name,
                'mae': round(mae, 2),
                'rmse': round(rmse, 2),
                'mape': round(mape, 2),
                'is_best': False
            })
    
    if results:
        results_df = pd.DataFrame(results)
        best_idx = results_df['mape'].idxmin()
        results_df.loc[best_idx, 'is_best'] = True
        best_model = results_df.iloc[best_idx]['model']
    else:
        results_df = pd.DataFrame()
        best_model = "No model available"
    
    return {
        'train': train,
        'test': test,
        'predictions': predictions,
        'metrics': results_df.to_dict('records') if len(results_df) > 0 else [],
        'best_model': best_model
    }

# -----------------------------
# Plot Generation (Base64 for web display)
# -----------------------------
def plot_to_base64():
    """Generate forecast plot as base64 string"""
    global sales_data, forecast_results
    
    if sales_data is None:
        sales_data = generate_sales_data()
        forecast_results = run_forecasting(sales_data)
    
    train = forecast_results['train']
    test = forecast_results['test']
    predictions = forecast_results['predictions']
    best_model_name = forecast_results['best_model']
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot historical data
    ax.plot(train.index, train['Sales'], label='Training Data', color='gray', linewidth=1.5, alpha=0.7)
    ax.plot(test.index, test['Sales'], label='Actual Sales (Test)', color='black', linewidth=2.5, marker='o', markersize=6, zorder=10)
    
    # Plot best model
    if best_model_name in predictions:
        best_pred = predictions[best_model_name]
        best_mape = next((m['mape'] for m in forecast_results['metrics'] if m['model'] == best_model_name), 0)
        ax.plot(test.index, best_pred, label=f'{best_model_name} (MAPE: {best_mape:.1f}%)',
               color='blue', linewidth=2.5, linestyle='-')
    
    # Also show Holt-Winters if it's not the best
    if 'Holt-Winters Additive' in predictions and best_model_name != 'Holt-Winters Additive':
        hw_mape = next((m['mape'] for m in forecast_results['metrics'] if m['model'] == 'Holt-Winters Additive'), 0)
        ax.plot(test.index, predictions['Holt-Winters Additive'], 
               label=f'Holt-Winters Additive (MAPE: {hw_mape:.1f}%)',
               color='green', linewidth=2, linestyle='--', alpha=0.8)
    
    # Highlight December peaks
    for date in train.index:
        if date.month == 12:
            ax.axvline(x=date, color='gold', alpha=0.15, linewidth=10, zorder=0)
    for date in test.index:
        if date.month == 12:
            ax.axvline(x=date, color='gold', alpha=0.3, linewidth=10, zorder=0)
    
    ax.set_title(f'Retail Sales Forecast - Best Model: {best_model_name}\nDataset: Synthetic (36 months with December peaks)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sales ($)', fontsize=12)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

def generate_diagnostics_base64():
    """Generate diagnostics plots"""
    global sales_data
    if sales_data is None:
        sales_data = generate_sales_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time series with rolling mean
    ax1 = axes[0, 0]
    ax1.plot(sales_data.index, sales_data['Sales'], color='black', linewidth=1.5)
    rolling_mean = sales_data['Sales'].rolling(window=6).mean()
    ax1.plot(sales_data.index, rolling_mean, label='6-Month Rolling Mean', color='red', linestyle='--', linewidth=2)
    ax1.set_title('Sales Trend with Rolling Mean', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Sales ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Seasonal profile by month
    ax2 = axes[0, 1]
    monthly_avg = sales_data.groupby('Month')['Sales'].mean()
    colors = ['#1f77b4'] * 12
    colors[11] = '#e94560'  # December in red
    colors[6] = '#ffa500'    # July in orange
    ax2.bar(monthly_avg.index, monthly_avg.values, color=colors, edgecolor='black')
    ax2.set_title('Average Sales by Month (Seasonal Profile)', fontweight='bold', fontsize=12)
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Average Sales ($)')
    ax2.axhline(y=sales_data['Sales'].mean(), color='red', linestyle='--', 
                label=f'Annual Avg: ${sales_data["Sales"].mean():.0f}')
    ax2.legend()
    
    # Yearly comparison
    ax3 = axes[1, 0]
    yearly_total = sales_data.groupby('Year')['Sales'].sum()
    ax3.bar(yearly_total.index.astype(str), yearly_total.values, color='coral', edgecolor='black')
    ax3.set_title('Year-over-Year Sales Growth', fontweight='bold', fontsize=12)
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Total Sales ($)')
    for i, v in enumerate(yearly_total.values):
        ax3.text(i, v + 500, f'${v:,.0f}', ha='center', fontsize=10)
    
    # Sales distribution
    ax4 = axes[1, 1]
    ax4.hist(sales_data['Sales'], bins=15, color='purple', edgecolor='black', alpha=0.7)
    ax4.set_title('Sales Distribution', fontweight='bold', fontsize=12)
    ax4.set_xlabel('Sales ($)')
    ax4.set_ylabel('Frequency')
    ax4.axvline(sales_data['Sales'].mean(), color='red', linestyle='--', 
                label=f'Mean: ${sales_data["Sales"].mean():.0f}')
    ax4.axvline(sales_data['Sales'].median(), color='green', linestyle='--', 
                label=f'Median: ${sales_data["Sales"].median():.0f}')
    ax4.legend()
    
    plt.tight_layout()
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode()
    plt.close()
    
    return image_base64

# -----------------------------
# Flask Routes
# -----------------------------
@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/data')
def get_data():
    """Get raw sales data"""
    global sales_data
    if sales_data is None:
        sales_data = generate_sales_data()
    
    return jsonify({
        'dates': sales_data.index.strftime('%Y-%m-%d').tolist(),
        'sales': sales_data['Sales'].tolist(),
        'summary': {
            'total_sales': float(sales_data['Sales'].sum()),
            'avg_sales': float(sales_data['Sales'].mean()),
            'max_sales': float(sales_data['Sales'].max()),
            'min_sales': float(sales_data['Sales'].min()),
            'period_start': sales_data.index.min().strftime('%Y-%m-%d'),
            'period_end': sales_data.index.max().strftime('%Y-%m-%d'),
            'months': len(sales_data)
        }
    })

@app.route('/api/forecast')
def get_forecast():
    """Get forecast results"""
    global forecast_results
    if forecast_results is None:
        sales_data = generate_sales_data()
        forecast_results = run_forecasting(sales_data)
    
    test = forecast_results['test']
    predictions = forecast_results['predictions']
    best_model_name = forecast_results['best_model']
    
    best_predictions = predictions[best_model_name].tolist() if best_model_name in predictions else []
    
    # Business insights
    if len(test) > 0 and len(best_predictions) > 0:
        avg_error = float(np.mean(np.abs(test['Sales'].values - best_predictions)))
        safety_stock = avg_error * 1.645
    else:
        avg_error = 0
        safety_stock = 0
    
    return jsonify({
        'test_dates': test.index.strftime('%Y-%m-%d').tolist(),
        'actual_sales': test['Sales'].tolist(),
        'best_model': best_model_name,
        'predictions': best_predictions,
        'metrics': forecast_results['metrics'],
        'business_kpis': {
            'avg_forecast_error': avg_error,
            'safety_stock_95': safety_stock,
            'data_quality': 'Good' if len(sales_data) >= 24 else 'Limited',
            'forecast_horizon': len(test)
        }
    })

@app.route('/api/plot/forecast')
def get_forecast_plot():
    """Get forecast plot as base64"""
    image_base64 = plot_to_base64()
    return jsonify({'image': image_base64})

@app.route('/api/plot/diagnostics')
def get_diagnostics_plot():
    """Get diagnostics plot as base64"""
    image_base64 = generate_diagnostics_base64()
    return jsonify({'image': image_base64})

@app.route('/api/refresh')
def refresh_data():
    """Regenerate data and forecasts"""
    global sales_data, forecast_results
    sales_data = generate_sales_data()
    forecast_results = run_forecasting(sales_data)
    return jsonify({'status': 'success', 'message': 'Data refreshed successfully'})

# -----------------------------
# Create Templates
# -----------------------------
def create_templates():
    """Create HTML templates"""
    template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Retail Sales Forecasting Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-3.0.1.min.js" charset="utf-8"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white;
            border-radius: 20px;
            padding: 30px;
            margin-bottom: 25px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
            text-align: center;
        }
        .header h1 { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p { color: #666; font-size: 1.1em; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 20px;
            margin-bottom: 25px;
        }
        .stat-card {
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.3s ease;
        }
        .stat-card:hover { transform: translateY(-5px); }
        .stat-card h3 { color: #667eea; font-size: 0.85em; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; }
        .stat-card .value { font-size: 2em; font-weight: bold; color: #333; }
        .stat-card .unit { font-size: 0.8em; color: #666; }
        .chart-container {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        .chart-container h2 { color: #333; margin-bottom: 15px; font-size: 1.3em; }
        .metrics-table {
            background: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 25px;
            overflow-x: auto;
        }
        .metrics-table h2 { color: #333; margin-bottom: 15px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #e0e0e0; }
        th { background: #667eea; color: white; font-weight: 600; border-radius: 10px; }
        tr:hover { background: #f5f5f5; }
        .best-model { background: #d4edda !important; font-weight: bold; }
        .btn-refresh {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            margin-top: 15px;
            transition: all 0.3s ease;
        }
        .btn-refresh:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(102,126,234,0.4); }
        .insights {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 25px;
        }
        .insights h3 { margin-bottom: 15px; font-size: 1.2em; }
        .insights ul { list-style: none; padding-left: 0; }
        .insights li { padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.2); }
        .insights li:last-child { border-bottom: none; }
        .badge {
            display: inline-block;
            background: #e94560;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.8em;
            margin-left: 10px;
        }
        .loading { text-align: center; padding: 50px; }
        .footer {
            text-align: center;
            color: white;
            padding: 20px;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏪 Retail Sales Forecasting Dashboard</h1>
            <p>Advanced Time Series Analysis for Inventory Planning</p>
            <button class="btn-refresh" onclick="refreshData()">🔄 Refresh Data & Re-forecast</button>
        </div>
        
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card"><h3>Total Sales</h3><div class="value" id="totalSales">--</div></div>
            <div class="stat-card"><h3>Monthly Average</h3><div class="value" id="avgSales">--</div></div>
            <div class="stat-card"><h3>Peak Month</h3><div class="value" id="peakSales">--</div></div>
            <div class="stat-card"><h3>Data Period</h3><div class="value" id="dataPeriod" style="font-size: 1.2em;">--</div></div>
        </div>
        
        <div class="chart-container">
            <h2>📊 Forecast vs Actual Sales <span class="badge">Best Model Highlighted</span></h2>
            <div id="forecastPlot" style="height: 500px;"></div>
        </div>
        
        <div class="chart-container">
            <h2>📈 Data Diagnostics & Seasonality Analysis</h2>
            <div id="diagnosticsPlot" style="min-height: 500px;"></div>
        </div>
        
        <div class="metrics-table">
            <h2>📋 Model Performance Comparison</h2>
            <div id="metricsTable">Loading...</div>
        </div>
        
        <div class="insights" id="insights">
            <h3>💡 Actionable Business Insights</h3>
            <div id="insightsContent">Loading insights...</div>
        </div>
        <div class="footer">
            <p>📊 Forecast models: Naive | Moving Average | SES | Holt's Linear | Holt-Winters | Damped Trend</p>
        </div>
    </div>
    
    <script>
        async function loadDashboard() {
            await loadData();
            await loadForecastPlot();
            await loadDiagnosticsPlot();
            await loadMetrics();
        }
        
        async function loadData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                document.getElementById('totalSales').innerHTML = '$' + data.summary.total_sales.toLocaleString();
                document.getElementById('avgSales').innerHTML = '$' + Math.round(data.summary.avg_sales).toLocaleString();
                document.getElementById('peakSales').innerHTML = '$' + data.summary.max_sales.toLocaleString();
                document.getElementById('dataPeriod').innerHTML = data.summary.period_start + '<br>to<br>' + data.summary.period_end;
            } catch(e) { console.error('Error loading data:', e); }
        }
        
        async function loadForecastPlot() {
            try {
                const forecastRes = await fetch('/api/forecast');
                const forecastData = await forecastRes.json();
                const salesRes = await fetch('/api/data');
                const salesData = await salesRes.json();
                
                const trace1 = {
                    x: salesData.dates,
                    y: salesData.sales,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Historical Sales',
                    line: { color: '#888', width: 2 },
                    opacity: 0.7
                };
                
                const trace2 = {
                    x: forecastData.test_dates,
                    y: forecastData.actual_sales,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Actual (Test Period)',
                    line: { color: 'black', width: 3 },
                    marker: { size: 8, color: 'black', symbol: 'circle' }
                };
                
                const trace3 = {
                    x: forecastData.test_dates,
                    y: forecastData.predictions,
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: `📈 Forecast: ${forecastData.best_model}`,
                    line: { color: '#e94560', width: 3, dash: 'dash' },
                    marker: { size: 6, color: '#e94560', symbol: 'diamond' }
                };
                
                const layout = {
                    title: `🎯 Best Model: ${forecastData.best_model} | MAPE: ${forecastData.metrics.find(m => m.model === forecastData.best_model)?.mape || '?'}%`,
                    xaxis: { title: 'Date', tickangle: -45, gridcolor: '#eee' },
                    yaxis: { title: 'Sales ($)', gridcolor: '#eee' },
                    hovermode: 'closest',
                    showlegend: true,
                    height: 500,
                    plot_bgcolor: 'white',
                    paper_bgcolor: 'white'
                };
                
                Plotly.newPlot('forecastPlot', [trace1, trace2, trace3], layout);
            } catch(e) { console.error('Error loading forecast plot:', e); }
        }
        
        async function loadDiagnosticsPlot() {
            try {
                const response = await fetch('/api/plot/diagnostics');
                const data = await response.json();
                const img = document.createElement('img');
                img.src = 'data:image/png;base64,' + data.image;
                img.style.width = '100%';
                img.style.borderRadius = '10px';
                img.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
                const container = document.getElementById('diagnosticsPlot');
                container.innerHTML = '';
                container.appendChild(img);
            } catch(e) { console.error('Error loading diagnostics:', e); }
        }
        
        async function loadMetrics() {
            try {
                const response = await fetch('/api/forecast');
                const data = await response.json();
                
                if (!data.metrics || data.metrics.length === 0) {
                    document.getElementById('metricsTable').innerHTML = '<p>No metrics available</p>';
                    return;
                }
                
                let html = '<table><thead><tr><th>Model</th><th>MAE ($)</th><th>RMSE ($)</th><th>MAPE (%)</th><th>Status</th></tr></thead><tbody>';
                data.metrics.forEach(metric => {
                    html += `<tr ${metric.is_best ? 'class="best-model"' : ''}>`;
                    html += `<td><strong>${metric.model}</strong>${metric.is_best ? ' 🏆' : ''}</td>`;
                    html += `<td>$${metric.mae.toLocaleString()}</td>`;
                    html += `<td>$${metric.rmse.toLocaleString()}</td>`;
                    html += `<td><strong>${metric.mape}%</strong></td>`;
                    html += `<td>${metric.is_best ? '✓ Best Performing' : ''}</td>`;
                    html += `</tr>`;
                });
                html += '</tbody></table>';
                document.getElementById('metricsTable').innerHTML = html;
                
                const bestModel = data.metrics.find(m => m.is_best);
                const baseline = data.metrics.find(m => m.model === 'Naive (Baseline)');
                const improvement = baseline ? (baseline.mape - bestModel.mape).toFixed(1) : 'N/A';
                
                const insightsHtml = `
                    <ul>
                        <li>🎯 <strong>Best Model:</strong> ${bestModel.model} achieves <strong>${bestModel.mape}% MAPE</strong> forecast accuracy</li>
                        <li>📦 <strong>Recommended Safety Stock (95% confidence):</strong> $${data.business_kpis.safety_stock_95.toLocaleString()}</li>
                        <li>🎄 <strong>December Peak Alert:</strong> Expect significant holiday sales increase - Order inventory by September</li>
                        <li>☀️ <strong>July Promotion Opportunity:</strong> Mid-year sales lift detected - Plan promotions accordingly</li>
                        <li>💰 <strong>Forecast Improvement:</strong> Using ${bestModel.model} reduces forecast error by <strong>${improvement}%</strong> compared to Naive baseline</li>
                        <li>📊 <strong>Data Quality:</strong> ${data.business_kpis.data_quality} | <strong>${data.business_kpis.forecast_horizon}</strong> months forecast horizon</li>
                    </ul>
                `;
                document.getElementById('insightsContent').innerHTML = insightsHtml;
            } catch(e) { console.error('Error loading metrics:', e); }
        }
        
        async function refreshData() {
            const btn = document.querySelector('.btn-refresh');
            btn.textContent = '🔄 Refreshing...';
            btn.disabled = true;
            try {
                const response = await fetch('/api/refresh');
                const result = await response.json();
                await loadDashboard();
                btn.textContent = '✅ ' + result.message;
                setTimeout(() => {
                    btn.textContent = '🔄 Refresh Data & Re-forecast';
                    btn.disabled = false;
                }, 2000);
            } catch(e) {
                console.error('Error refreshing:', e);
                btn.textContent = '❌ Error';
                setTimeout(() => {
                    btn.textContent = '🔄 Refresh Data & Re-forecast';
                    btn.disabled = false;
                }, 2000);
            }
        }
        
        // Load dashboard on page load
        loadDashboard();
    </script>
</body>
</html>'''
    
    with open(os.path.join(template_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ Templates created successfully")

# -----------------------------
# Main Entry Point
# -----------------------------
def open_browser():
    """Open browser after a short delay"""
    time.sleep(1.5)
    webbrowser.open('http://127.0.0.1:5000')

if __name__ == '__main__':
    print("=" * 70)
    print("🏪 RETAIL SALES FORECASTING DASHBOARD")
    print("=" * 70)
    
    # Create templates
    create_templates()
    
    # Generate initial data
    print("\n📊 Generating sales data and training models...")
    sales_data = generate_sales_data()
    forecast_results = run_forecasting(sales_data)
    print(f"✅ Data generated: {len(sales_data)} months of sales data")
    print(f"✅ Best model: {forecast_results['best_model']}")
    
    print("\n" + "=" * 70)
    print("🌐 STARTING WEB SERVER")
    print("=" * 70)
    print("\n👉 Opening browser at: http://127.0.0.1:5000")
    print("👉 If browser doesn't open, manually navigate to the URL above")
    print("👉 Press CTRL+C to stop the server")
    print("=" * 70)
    
    # Open browser automatically
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run Flask app on localhost only (no external access, avoids permission issues)
    app.run(debug=False, host='127.0.0.1', port=5000)
