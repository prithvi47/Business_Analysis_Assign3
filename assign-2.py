import numpy as np
import pandas as pd
from flask import Flask, render_template_string, jsonify, request
import plotly
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from jinja2 import Template
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24).hex()

# -------------------- Data Loading (real Kaggle dataset if available) --------------------
DATA_FILE = 'agriculture.csv'

def load_real_data():
    if os.path.exists(DATA_FILE):
        try:
            df = pd.read_csv(DATA_FILE)
            # Ensure required columns exist, otherwise generate
            required = ['date', 'temperature', 'humidity', 'soil_moisture', 'rainfall',
                        'crop_yield', 'field', 'crop_type', 'ndvi', 'pest_risk',
                        'disease_risk', 'water_stress', 'equipment_hours', 'co2_emission']
            if all(col in df.columns for col in required):
                return df
        except:
            pass
    return None

REAL_DATA_AVAILABLE = load_real_data() is not None

def generate_sample_data(farm=None, start_date=None, end_date=None, crops=None):
    real_df = load_real_data()
    if real_df is not None:
        df = real_df.copy()
        if crops:
            df = df[df['crop_type'].isin(crops)]
        if farm and 'farm' in df.columns:
            df = df[df['farm'] == farm]
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        np.random.seed(42 + hash(farm) % 100 if farm else 42)
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        n = len(dates)
        df = pd.DataFrame({
            'date': dates,
            'temperature': np.random.normal(25, 5, n),
            'humidity': np.random.normal(65, 10, n),
            'soil_moisture': np.random.normal(70, 15, n),
            'rainfall': np.random.exponential(5, n),
            'crop_yield': np.random.normal(1200, 200, n),
            'field': np.random.choice(['Field A', 'Field B', 'Field C', 'Field D'], n),
            'crop_type': np.random.choice(['Corn', 'Wheat', 'Soybeans'], n),
            'ndvi': np.random.uniform(0.5, 0.9, n),
            'pest_risk': np.random.uniform(0, 1, n),
            'disease_risk': np.random.uniform(0, 1, n),
            'water_stress': np.random.uniform(0, 1, n),
            'equipment_hours': np.random.randint(0, 500, n),
            'co2_emission': np.random.uniform(10, 50, n)
        })
        if crops:
            df = df[df['crop_type'].isin(crops)]
        return df

# -------------------- Helper Functions --------------------
def get_kpi_data(filters=None):
    df = generate_sample_data(**filters) if filters else generate_sample_data()
    return {
        'total_yield': f"{df['crop_yield'].sum():,.0f} kg",
        'avg_temp': f"{df['temperature'].mean():.1f} °C",
        'avg_soil_moisture': f"{df['soil_moisture'].mean():.0f}%",
        'total_rainfall': f"{df['rainfall'].sum():,.0f} mm",
        'co2_footprint': f"{df['co2_emission'].mean():.1f} kg/ha"
    }

def detect_anomalies():
    df = generate_sample_data().groupby('field').agg({
        'temperature': 'mean',
        'humidity': 'mean',
        'soil_moisture': 'mean',
        'crop_yield': 'mean',
        'equipment_hours': 'mean'
    }).reset_index()
    features = df[['temperature', 'humidity', 'soil_moisture', 'crop_yield', 'equipment_hours']]
    iso_forest = IsolationForest(contamination=0.2, random_state=42)
    df['anomaly'] = iso_forest.fit_predict(features)
    anomalies = df[df['anomaly'] == -1]['field'].tolist()
    return anomalies

def forecast_yield(days=30):
    df = generate_sample_data().set_index('date')['crop_yield']
    model = sm.tsa.ARIMA(df, order=(1,1,1)).fit()
    forecast = model.forecast(steps=days)
    return forecast.tolist()

def get_ai_recommendation():
    anomalies = detect_anomalies()
    if anomalies:
        return f"⚠️ Anomaly detected in {', '.join(anomalies)}. Check sensors/equipment."
    else:
        return "✅ All fields operating normally. Optimal irrigation schedule."

def predict_maintenance():
    fields = ['Field A', 'Field B', 'Field C', 'Field D']
    hours = [np.random.randint(200, 500) for _ in fields]
    maintenance = ['Due soon' if h > 400 else 'OK' for h in hours]
    return dict(zip(fields, zip(hours, maintenance)))

# -------------------- Professional Logo (SVG) --------------------
LOGO_SVG = '''
<svg width="180" height="50" viewBox="0 0 180 50" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect width="180" height="50" rx="10" fill="white"/>
    <path d="M25 25 L35 15 L45 25 L40 30 L35 25 L30 30 L25 25" fill="#01B763" stroke="#005B31" stroke-width="2"/>
    <circle cx="35" cy="20" r="3" fill="#E74C3C"/>
    <text x="55" y="32" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#005B31">PK Farm</text>
    <text x="55" y="45" font-family="Arial, sans-serif" font-size="10" fill="#01B763">smart agriculture</text>
</svg>
'''

# -------------------- Context Processor --------------------
@app.context_processor
def inject_globals():
    return {
        'now': datetime.now(),
        'timedelta': timedelta,
        'logo_svg': LOGO_SVG
    }

# -------------------- Base Template --------------------
BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Agriculture</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: #f5f9f5;
            color: #1e2e1e;
            transition: background 0.3s;
        }
        body.dark-mode {
            background: #1e2e1e;
            color: #f5f9f5;
        }
        .app-container { display: flex; min-height: 100vh; }
        .sidebar {
            width: 280px;
            background: linear-gradient(180deg, #005B31 0%, #01B763 100%);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            box-shadow: 4px 0 20px rgba(0,60,0,0.3);
            animation: slideInLeft 0.5s ease;
            color: white;
        }
        @keyframes slideInLeft {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        .logo svg { filter: drop-shadow(0 4px 6px rgba(0,40,0,0.5)); }
        .profile-card {
            background: rgba(255,255,255,0.1);
            border-radius: 20px;
            padding: 1rem;
            display: flex;
            align-items: center;
            gap: 1rem;
            backdrop-filter: blur(5px);
            border: 1px solid rgba(255,255,255,0.2);
            transition: transform 0.2s;
        }
        .profile-card:hover { transform: scale(1.02); box-shadow: 0 8px 20px rgba(0,80,0,0.3); }
        .avatar {
            width: 50px; height: 50px;
            background: linear-gradient(135deg, #01B763, #005B31);
            border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.5rem; box-shadow: 0 4px 10px rgba(0,80,0,0.4);
        }
        .profile-info .name { font-weight: 600; font-size: 1.1rem; }
        .profile-info .role { font-size: 0.85rem; opacity: 0.7; }
        .nav-menu { display: flex; flex-direction: column; gap: 0.5rem; }
        .nav-item {
            display: flex; align-items: center; gap: 0.75rem;
            padding: 0.75rem 1rem; border-radius: 12px;
            color: white; text-decoration: none;
            transition: all 0.2s; border: 1px solid transparent;
            font-weight: 500;
        }
        .nav-item:hover {
            background: rgba(255,255,255,0.15);
            transform: translateX(5px);
            border-color: rgba(255,255,255,0.2);
        }
        .nav-item.active {
            background: white;
            color: #005B31;
            border: none; box-shadow: 0 4px 15px rgba(255,255,255,0.3);
        }
        .filters {
            background: rgba(255,255,255,0.05);
            border-radius: 16px; padding: 1rem;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .filters h4 { margin-bottom: 1rem; opacity: 0.8; font-weight: 500; }
        .filter-select, .date-input {
            width: 100%; padding: 0.6rem; border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.3);
            background: rgba(0,0,0,0.1); color: white;
            margin-bottom: 0.5rem; transition: all 0.2s;
        }
        .filter-select:hover, .date-input:hover { border-color: #01B763; }
        .date-range { display: flex; gap: 0.5rem; align-items: center; margin: 0.5rem 0; }
        .crop-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
        .chip {
            background: rgba(255,255,255,0.1); padding: 0.3rem 1rem;
            border-radius: 30px; font-size: 0.85rem; cursor: pointer;
            transition: all 0.2s; border: 1px solid transparent;
        }
        .chip:hover { background: rgba(1,183,99,0.4); transform: translateY(-2px); }
        .chip.selected { background: #01B763; color: #005B31; font-weight: 600; box-shadow: 0 4px 10px rgba(1,183,99,0.4); }
        .refresh-btn {
            background: linear-gradient(90deg, #01B763, #005B31);
            border: none; padding: 0.75rem; border-radius: 12px;
            color: white; font-weight: 600; cursor: pointer;
            transition: all 0.2s; animation: pulseGreen 2s infinite;
            box-shadow: 0 4px 15px rgba(1,183,99,0.3);
        }
        .refresh-btn:hover { transform: scale(1.02); box-shadow: 0 6px 20px rgba(1,183,99,0.5); }
        @keyframes pulseGreen {
            0% { box-shadow: 0 0 0 0 rgba(1,183,99,0.7); }
            70% { box-shadow: 0 0 0 10px rgba(1,183,99,0); }
            100% { box-shadow: 0 0 0 0 rgba(1,183,99,0); }
        }
        .update-time { text-align: center; font-size: 0.8rem; opacity: 0.6; margin-top: 0.5rem; }
        .main-content {
            flex: 1; padding: 2rem; overflow-y: auto;
            animation: fadeIn 0.5s ease;
        }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .theme-toggle {
            position: fixed; bottom: 20px; right: 20px;
            background: rgba(255,255,255,0.2); backdrop-filter: blur(10px);
            width: 50px; height: 50px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            cursor: pointer; border: 1px solid rgba(255,255,255,0.3);
            z-index: 1000; transition: all 0.3s;
            color: white;
        }
        .theme-toggle:hover { transform: rotate(15deg) scale(1.1); background: #01B763; }
        .spinner {
            border: 4px solid rgba(1,183,99,0.2); border-top: 4px solid #01B763;
            border-radius: 50%; width: 40px; height: 40px;
            animation: spin 1s linear infinite; margin: 20px auto;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-bottom: 1.5rem; }
        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-bottom: 1.5rem; }
        .grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1.5rem; margin-bottom: 1.5rem; }
        .card {
            background: white; border-radius: 24px; padding: 1.5rem;
            border: 1px solid #c8e6c9; box-shadow: 0 10px 30px -10px rgba(0,80,0,0.1);
            transition: all 0.3s; animation: cardAppear 0.5s ease backwards;
        }
        body.dark-mode .card { background: #2e3b2e; border-color: #01B763; }
        .card:hover { transform: translateY(-5px); border-color: #01B763; box-shadow: 0 15px 30px rgba(1,183,99,0.15); }
        .card h3 { color: #005B31; margin-bottom: 1rem; font-weight: 600; }
        body.dark-mode .card h3 { color: #01B763; }
        .kpi-value { font-size: 2rem; font-weight: 700; color: #005B31; }
        body.dark-mode .kpi-value { color: #01B763; }
        .trend {
            font-size: 0.85rem; padding: 0.25rem 0.75rem; border-radius: 30px;
            display: inline-block;
        }
        .trend.up { background: #e8f5e9; color: #005B31; }
        .trend.down { background: #ffebee; color: #E74C3C; }
        .welcome-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 2rem;
        }
        .welcome-header h1 { font-size: 2rem; color: #005B31; }
        .date-badge {
            background: #01B763; color: white; padding: 0.5rem 1rem;
            border-radius: 30px; font-weight: 500;
        }
        .weather-card {
            display: flex; gap: 2rem; align-items: center;
        }
        .weather-item { text-align: center; }
        .weather-item .value { font-size: 1.8rem; font-weight: 700; color: #005B31; }
        .weather-item .label { color: #4a6b5a; }
        .soil-moisture-stats { display: flex; justify-content: space-around; }
        .stat-number { font-size: 1.8rem; font-weight: 700; color: #01B763; }
        .growth-timeline {
            display: flex; gap: 1rem; justify-content: space-between;
        }
        .stage {
            background: #e8f5e9; padding: 1rem; border-radius: 12px; flex: 1;
            text-align: center;
        }
        .stage .phase { font-weight: 600; color: #005B31; }
        .stage .week { color: #01B763; }
        .water-level {
            display: flex; gap: 0.5rem; flex-wrap: wrap;
        }
        .water-tag {
            background: #e8f5e9; padding: 0.5rem 1rem; border-radius: 30px;
            color: #005B31; font-weight: 500;
        }
        .harvest-item {
            display: flex; justify-content: space-between;
            padding: 0.75rem 0; border-bottom: 1px solid #c8e6c9;
        }
        .harvest-item:last-child { border-bottom: none; }
        .btn {
            padding: 0.5rem 1.5rem; border-radius: 30px; border: none;
            font-weight: 600; cursor: pointer; transition: all 0.2s;
        }
        .btn-primary { background: #01B763; color: white; }
        .btn-primary:hover { background: #005B31; transform: scale(1.05); }
        .btn-secondary { background: #E74C3C; color: white; }
        .btn-secondary:hover { background: #c0392b; }
        .analytics-desc {
            background: #e8f5e9; padding: 1rem; border-radius: 12px;
            margin-bottom: 1.5rem; color: #005B31;
        }
        @media (max-width: 1024px) {
            .grid-4 { grid-template-columns: repeat(2, 1fr); }
            .grid-3 { grid-template-columns: repeat(2, 1fr); }
        }
        @media (max-width: 768px) {
            .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="app-container">
        <aside class="sidebar">
            <div class="logo">
                {{ logo_svg|safe }}
            </div>
            <div class="profile-card">
                <div class="avatar">👨‍🌾</div>
                <div class="profile-info">
                    <div class="name">Prithvi Raj</div>
                    <div class="role">Owner</div>
                </div>
            </div>
            <nav class="nav-menu">
                <a href="/" class="nav-item {{ 'active' if request.path == '/' else '' }}"><span>🌾</span> Dashboard</a>
                <a href="/analytics" class="nav-item {{ 'active' if request.path == '/analytics' else '' }}"><span>📊</span> Analytics</a>
                <a href="/crop-health" class="nav-item {{ 'active' if request.path == '/crop-health' else '' }}"><span>🌱</span> Crop Health</a>
                <a href="/irrigation" class="nav-item {{ 'active' if request.path == '/irrigation' else '' }}"><span>💧</span> Irrigation</a>
                <a href="/forecasting" class="nav-item {{ 'active' if request.path == '/forecasting' else '' }}"><span>📈</span> Forecasting</a>
                <a href="/maintenance" class="nav-item {{ 'active' if request.path == '/maintenance' else '' }}"><span>🔧</span> Maintenance</a>
                <a href="/carbon" class="nav-item {{ 'active' if request.path == '/carbon' else '' }}"><span>🌍</span> Carbon</a>
                <a href="/energy" class="nav-item {{ 'active' if request.path == '/energy' else '' }}"><span>⚡</span> Energy</a>
                <a href="/settings" class="nav-item {{ 'active' if request.path == '/settings' else '' }}"><span>⚙️</span> Settings</a>
            </nav>
            <div class="filters">
                <h4>🔍 Filters</h4>
                <select id="farm-select" class="filter-select">
                    <option>Green Valley Farm</option>
                    <option>Sunrise Fields</option>
                    <option>Mountain View Ranch</option>
                </select>
                <div class="date-range">
                    <input type="date" id="start-date" class="date-input" value="{{ (now - timedelta(days=30)).strftime('%Y-%m-%d') }}">
                    <span>—</span>
                    <input type="date" id="end-date" class="date-input" value="{{ now.strftime('%Y-%m-%d') }}">
                </div>
                <div class="crop-chips" id="crop-chips">
                    <span class="chip" data-crop="Corn">Corn</span>
                    <span class="chip" data-crop="Wheat">Wheat</span>
                    <span class="chip" data-crop="Soybeans">Soybeans</span>
                </div>
            </div>
            <button id="refresh-btn" class="refresh-btn">🔄 Refresh Data</button>
            <div class="update-time">Last update: <span id="last-update">{{ now.strftime('%H:%M:%S') }}</span></div>
        </aside>
        <main class="main-content" id="main-content">
            {{ page_content | safe }}
        </main>
    </div>
    <div class="theme-toggle" id="theme-toggle">🌓</div>
    <div id="notification-area"></div>

    <script>
        let selectedCrops = ['Corn', 'Wheat'];
        document.querySelectorAll('.chip').forEach(chip => {
            const crop = chip.dataset.crop;
            if (selectedCrops.includes(crop)) chip.classList.add('selected');
            chip.addEventListener('click', function() {
                const crop = this.dataset.crop;
                if (selectedCrops.includes(crop)) {
                    selectedCrops = selectedCrops.filter(c => c !== crop);
                    this.classList.remove('selected');
                } else {
                    selectedCrops.push(crop);
                    this.classList.add('selected');
                }
                if (window.refreshPageData) window.refreshPageData();
            });
        });

        document.getElementById('refresh-btn').addEventListener('click', function() {
            showSpinner();
            if (window.refreshPageData) window.refreshPageData();
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            hideSpinner();
        });

        document.getElementById('farm-select').addEventListener('change', function() {
            if (window.refreshPageData) window.refreshPageData();
        });
        document.getElementById('start-date').addEventListener('change', function() {
            if (window.refreshPageData) window.refreshPageData();
        });
        document.getElementById('end-date').addEventListener('change', function() {
            if (window.refreshPageData) window.refreshPageData();
        });

        function showSpinner() {
            let spinner = document.getElementById('global-spinner');
            if (!spinner) {
                spinner = document.createElement('div');
                spinner.id = 'global-spinner';
                spinner.className = 'spinner';
                spinner.style.position = 'fixed';
                spinner.style.top = '50%';
                spinner.style.left = '50%';
                spinner.style.transform = 'translate(-50%, -50%)';
                spinner.style.zIndex = '3000';
                document.body.appendChild(spinner);
            }
        }
        function hideSpinner() {
            const spinner = document.getElementById('global-spinner');
            if (spinner) spinner.remove();
        }

        function showNotification(message, type = 'info') {
            const area = document.getElementById('notification-area');
            const notif = document.createElement('div');
            notif.className = 'notification';
            notif.style.borderLeftColor = type === 'error' ? '#E74C3C' : '#01B763';
            notif.innerHTML = message;
            area.appendChild(notif);
            setTimeout(() => notif.remove(), 5000);
        }

        function updateSensors() {
            fetch('/api/sensor-data')
                .then(res => res.json())
                .then(data => {
                    document.querySelectorAll('.sensor-value').forEach(el => {
                        const sensor = el.dataset.sensor;
                        if (data[sensor] !== undefined) {
                            let unit = '';
                            if (sensor === 'temperature') unit = '°C';
                            else if (sensor === 'humidity') unit = '%';
                            else if (sensor === 'light') unit = ' lux';
                            else unit = '%';
                            el.textContent = data[sensor] + unit;
                            const progress = el.closest('.sensor-item')?.querySelector('.progress');
                            if (progress) {
                                let percent = data[sensor];
                                if (sensor === 'temperature') percent = (data[sensor] / 50) * 100;
                                if (sensor === 'light') percent = (data[sensor] / 1000) * 100;
                                progress.style.width = Math.min(percent, 100) + '%';
                            }
                        }
                    });
                });
        }
        setInterval(updateSensors, 30000);

        setInterval(() => {
            fetch('/api/anomalies')
                .then(res => res.json())
                .then(data => {
                    if (data.anomalies.length > 0) {
                        showNotification('⚠️ Anomaly detected in ' + data.anomalies.join(', '), 'error');
                    }
                });
        }, 60000);

        document.getElementById('theme-toggle').addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
        });
    </script>
</body>
</html>
'''

# -------------------- Page Templates --------------------
DASHBOARD_PAGE = '''
<div class="welcome-header">
    <h1>Welcome back, Prithvi!</h1>
    <div class="date-badge">{{ now.strftime('%d %B, %Y') }}</div>
</div>

<div class="grid-2">
    <!-- Weather Card -->
    <div class="card">
        <h3>☀️ Weather</h3>
        <div class="weather-card">
            <div class="weather-item">
                <div class="value">4.6 cm</div>
                <div class="label">Height</div>
            </div>
            <div class="weather-item">
                <div class="value">+22 °C</div>
                <div class="label">Soil temp</div>
            </div>
            <div class="weather-item">
                <div class="value">0 mm</div>
                <div class="label">Precipitation</div>
            </div>
        </div>
    </div>
    <!-- Soil Moisture Card -->
    <div class="card">
        <h3>💧 Soil moisture</h3>
        <div class="soil-moisture-stats">
            <div><span class="stat-number">15</span> Fields</div>
            <div><span class="stat-number">32</span> Fields</div>
            <div><span class="stat-number">28</span> Fields</div>
        </div>
    </div>
</div>

<div class="grid-3">
    <!-- Plant growth activity -->
    <div class="card">
        <h3>🌱 Plant growth activity</h3>
        <div class="growth-timeline">
            <div class="stage"><span class="phase">Seed phase</span><br><span class="week">Week 1</span></div>
            <div class="stage"><span class="phase">Vegetation</span><br><span class="week">Week 2</span></div>
            <div class="stage"><span class="phase">Final growth</span><br><span class="week">Week 3</span></div>
        </div>
    </div>
    <!-- Water level -->
    <div class="card">
        <h3>💧 Water level</h3>
        <div class="water-level">
            <span class="water-tag">Today</span>
            <span class="water-tag">20 Jan</span>
            <span class="water-tag">45 Days</span>
            <span class="water-tag">0 Days</span>
            <span class="water-tag">Nursery</span>
            <span class="water-tag">Seeding</span>
            <span class="water-tag">Harvest</span>
        </div>
    </div>
    <!-- Harvest schedule -->
    <div class="card">
        <h3>📅 Harvest schedule</h3>
        <div class="harvest-item">
            <span>Rundofase</span>
            <span>02 Feb 2023, 8:00 pm</span>
        </div>
        <div class="harvest-item">
            <span>Rapid Harvest & Co.</span>
            <span>06 Feb 2023, 6:30 pm</span>
        </div>
        <div class="harvest-item">
            <span>J-Texon</span>
            <span>10 Feb 2023, 9:00 am</span>
        </div>
        <button class="btn btn-primary" style="margin-top:1rem;">Ask a question</button>
    </div>
</div>

<!-- KPI Cards -->
<div class="grid-4">
    <div class="card"><h3>🌾 Total Yield</h3><div class="kpi-value">{{ kpi.total_yield }}</div><span class="trend up">↑ 8.2%</span></div>
    <div class="card"><h3>🌡️ Avg Temp</h3><div class="kpi-value">{{ kpi.avg_temp }}</div><span class="trend up">↑ 1.5°C</span></div>
    <div class="card"><h3>💧 Soil Moisture</h3><div class="kpi-value">{{ kpi.avg_soil_moisture }}</div><span class="trend down">↓ 3%</span></div>
    <div class="card"><h3>🌍 CO₂ Footprint</h3><div class="kpi-value">{{ kpi.co2_footprint }}</div><span class="trend down">↓ 5%</span></div>
</div>

<div class="ai-widget" id="ai-widget"><strong>🤖 AI Insight:</strong> <span id="ai-message">{{ ai_message }}</span></div>

<!-- Charts Row -->
<div class="grid-2">
    <div class="card"><h3>🛰️ Satellite View</h3><div id="satellite-heatmap"></div></div>
    <div class="card"><h3>📊 Sensors</h3><div class="sensor-list" id="sensor-list">
        <div class="sensor-item"><div class="sensor-header"><span>Soil Moisture</span><span class="sensor-value" data-sensor="soil_moisture">72%</span></div><div class="progress-bar"><div class="progress" style="width:72%"></div></div></div>
        <div class="sensor-item"><div class="sensor-header"><span>Temperature</span><span class="sensor-value" data-sensor="temperature">23°C</span></div><div class="progress-bar"><div class="progress" style="width:60%"></div></div></div>
        <div class="sensor-item"><div class="sensor-header"><span>Humidity</span><span class="sensor-value" data-sensor="humidity">65%</span></div><div class="progress-bar"><div class="progress" style="width:65%"></div></div></div>
        <div class="sensor-item"><div class="sensor-header"><span>Light</span><span class="sensor-value" data-sensor="light">850 lux</span></div><div class="progress-bar"><div class="progress" style="width:85%"></div></div></div>
    </div></div>
</div>

<div class="grid-2">
    <div class="card"><h3>📈 Yield Trends</h3><div id="yield-chart"></div></div>
    <div class="card"><h3>💧 Irrigation Schedule</h3>
        <table style="width:100%"><tr><th>Field</th><th>Last</th><th>Next</th><th>Moisture</th></tr>
        <tr><td>Field A</td><td>2026-02-14</td><td>2026-02-18</td><td>75%</td></tr>
        <tr><td>Field B</td><td>2026-02-13</td><td>2026-02-17</td><td>68%</td></tr>
        <tr><td>Field C</td><td>2026-02-15</td><td>2026-02-19</td><td>82%</td></tr>
        <tr><td>Field D</td><td>2026-02-12</td><td>2026-02-16</td><td>71%</td></tr>
        </table>
        <div style="background:#e8f5e9; padding:0.5rem; border-radius:12px; margin:1rem 0;"><strong>💡 AI:</strong> Field B needs irrigation.</div>
        <button id="activate-valves" class="btn btn-primary">🚰 Activate All Valves</button>
    </div>
</div>

<script>
    window.refreshPageData = function() {
        const farm = document.getElementById('farm-select').value;
        const startDate = document.getElementById('start-date').value;
        const endDate = document.getElementById('end-date').value;
        const params = new URLSearchParams({farm, start_date: startDate, end_date: endDate, crops: selectedCrops.join(',')});
        fetch('/api/dashboard-data?' + params).then(res=>res.json()).then(data => {
            document.querySelectorAll('.grid-4 .card .kpi-value')[0].textContent = data.kpi.total_yield;
            document.querySelectorAll('.grid-4 .card .kpi-value')[1].textContent = data.kpi.avg_temp;
            document.querySelectorAll('.grid-4 .card .kpi-value')[2].textContent = data.kpi.avg_soil_moisture;
            document.querySelectorAll('.grid-4 .card .kpi-value')[3].textContent = data.kpi.co2_footprint;
            document.getElementById('ai-message').textContent = data.ai_message;
            Plotly.react('yield-chart', data.yield_chart.data, data.yield_chart.layout);
            Plotly.react('satellite-heatmap', [{z: data.satellite_data.z, x: data.satellite_data.x, y: data.satellite_data.y, type: 'heatmap', colorscale: 'Greens'}], {margin:{t:0,b:0,l:0,r:0}, height:300});
        });
    };

    fetch('/api/chart-data').then(res=>res.json()).then(g=>Plotly.newPlot('yield-chart', g.data, g.layout));
    Plotly.newPlot('satellite-heatmap', [{z:[[1,0.8,0.7,0.9],[0.7,0.9,0.8,0.6],[0.8,0.7,0.9,0.8],[0.9,0.8,0.7,0.9]], x:['Field A','Field B','Field C','Field D'], y:['Field A','Field B','Field C','Field D'], type:'heatmap', colorscale:'Greens'}], {margin:{t:0,b:0,l:0,r:0}, height:300});
    document.getElementById('activate-valves').addEventListener('click', ()=>fetch('/api/activate-valves',{method:'POST'}).then(res=>res.json()).then(d=>alert(d.message)));
</script>
'''

ANALYTICS_PAGE = '''
<h1 style="color:#005B31;">📊 Analytics</h1>
<div class="analytics-desc">
    <strong>What you'll find here:</strong> Field clustering using K-Means (groups similar fields), anomaly detection (Isolation Forest) to spot unusual patterns, and trend analysis. Data updates based on your filters.
</div>
<div class="grid-2">
    <div class="card"><h3>Field Clustering (K-Means)</h3><div id="clustering-table"></div></div>
    <div class="card"><h3>Anomaly Detection</h3><div id="anomaly-chart"></div></div>
</div>
<script>
    window.refreshPageData = function() {
        const params = new URLSearchParams({farm:document.getElementById('farm-select').value, crops:selectedCrops.join(',')});
        fetch('/api/clustering-data?'+params).then(res=>res.json()).then(data => {
            let html = '<table style="width:100%"><tr><th>Field</th><th>Cluster</th><th>Avg Temp</th><th>Avg Humidity</th><th>Soil Moisture</th><th>Yield</th></tr>';
            data.forEach(f => html += `<tr><td>${f.field}</td><td>${f.cluster}</td><td>${f.temperature.toFixed(1)}°C</td><td>${f.humidity.toFixed(1)}%</td><td>${f.soil_moisture.toFixed(1)}%</td><td>${Math.round(f.crop_yield)} kg</td></tr>`);
            html += '</table>';
            document.getElementById('clustering-table').innerHTML = html;
        });
        fetch('/api/anomaly-chart-data?'+params).then(res=>res.json()).then(d=>Plotly.react('anomaly-chart', d.data, d.layout));
    };
    window.refreshPageData();
</script>
'''

CROP_HEALTH_PAGE = '''
<h1 style="color:#005B31;">🌱 Crop Health</h1>
<div class="grid-2">
    <div class="card"><h3>NDVI Index</h3><div id="ndvi-chart"></div></div>
    <div class="card"><h3>Health Radar</h3><div id="radar-chart"></div></div>
</div>
<div class="grid-2">
    <div class="card"><h3>Pest Alerts</h3><div id="alerts"><div style="background:#fff3e0; padding:1rem; border-radius:12px;">⚠️ Early blight in Field C</div><div style="background:#ffebee; padding:1rem; border-radius:12px; margin-top:0.5rem;">🚨 Grasshopper risk in Field A</div></div></div>
    <div class="card"><h3>Soil pH</h3><div id="ph-map"></div></div>
</div>
<script>
    var ndvi = [{x:['Field A','Field B','Field C','Field D'], y:[0.82,0.75,0.68,0.79], type:'bar', marker:{color:['#01B763','#ffb74d','#E74C3C','#66bb6a']}}];
    Plotly.newPlot('ndvi-chart', ndvi, {title:'NDVI'});
    var radar = [{type:'scatterpolar', r:[0.8,0.6,0.3,0.4,0.7], theta:['NDVI','Pest','Disease','Water','Nutrient'], fill:'toself'}];
    Plotly.newPlot('radar-chart', radar, {polar:{radialaxis:{range:[0,1]}}});
    var ph = [{z:[[6.5,6.8,7.0],[6.2,6.9,7.2],[6.0,6.5,6.8]], x:['Field A','Field B','Field C'], y:['North','Center','South'], type:'heatmap', colorscale:'Earth'}];
    Plotly.newPlot('ph-map', ph, {margin:{t:0,b:0,l:0,r:0}, height:250});
</script>
'''

IRRIGATION_PAGE = '''
<h1 style="color:#005B31;">💧 Irrigation</h1>
<div class="grid-2">
    <div class="card"><h3>Water Usage</h3><div id="water-chart"></div></div>
    <div class="card"><h3>Soil Moisture Gauge</h3><div id="gauge-chart"></div></div>
</div>
<div class="grid-2">
    <div class="card"><h3>Valve Control</h3><div style="display:flex; gap:1rem; flex-wrap:wrap;"><button class="valve-btn" data-field="A" style="padding:0.75rem; background:#01B763; border:none; border-radius:12px; color:white;">Field A (Open)</button><button class="valve-btn" data-field="B" style="padding:0.75rem; background:#01B763; border:none; border-radius:12px; color:white;">Field B (Open)</button><button class="valve-btn" data-field="C" style="padding:0.75rem; background:#E74C3C; border:none; border-radius:12px; color:white;">Field C (Closed)</button><button class="valve-btn" data-field="D" style="padding:0.75rem; background:#ffb74d; border:none; border-radius:12px;">Field D (Scheduled)</button></div><button id="activate-all" style="margin-top:1rem; padding:0.75rem; background:#01B763; color:white; border:none; border-radius:12px; width:100%;">Activate All</button></div>
    <div class="card"><h3>Weather Forecast</h3><div id="weather-chart"></div></div>
</div>
<script>
    var water = [{x:['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], y:[120,135,110,145,130,155,140], type:'scatter', fill:'tozeroy', line:{color:'#01B763'}}];
    Plotly.newPlot('water-chart', water, {title:'Water Usage'});
    var gauge = [{type:'indicator', mode:'gauge+number', value:72, title:{text:'Soil Moisture %'}, gauge:{axis:{range:[0,100]}, bar:{color:'#01B763'}}}];
    Plotly.newPlot('gauge-chart', gauge, {height:300});
    var weather = [{x:['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], y:[23,25,22,21,24,26,27], type:'scatter', mode:'lines+markers', line:{color:'#ffb74d'}}];
    Plotly.newPlot('weather-chart', weather, {title:'Temperature'});
    document.querySelectorAll('.valve-btn').forEach(b=>b.addEventListener('click', function(){fetch('/api/toggle-valve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({field:this.dataset.field})}).then(res=>res.json()).then(d=>alert(d.message));}));
    document.getElementById('activate-all').addEventListener('click', ()=>fetch('/api/activate-valves',{method:'POST'}).then(res=>res.json()).then(d=>alert(d.message)));
</script>
'''

FORECASTING_PAGE = '''
<h1 style="color:#005B31;">📈 Forecasting</h1>
<div class="grid-2">
    <div class="card"><h3>Yield Forecast (30 days)</h3><div id="forecast-chart"></div><button id="next-page-btn" class="btn btn-primary" style="margin-top:1rem;">Next →</button></div>
    <div class="card"><h3>Market Prices</h3><div id="price-chart"></div></div>
</div>
<div class="grid-2">
    <div class="card"><h3>Price Prediction</h3><div id="price-prediction"></div></div>
    <div class="card"><h3>Weather Impact</h3><div id="weather-impact"></div></div>
</div>
<script>
    let forecastPage = 1;
    window.refreshPageData = function() {
        const params = new URLSearchParams({farm:document.getElementById('farm-select').value, crops:selectedCrops.join(','), page:forecastPage});
        fetch('/api/forecast-data?'+params).then(res=>res.json()).then(d=>Plotly.react('forecast-chart', [{x:d.days, y:d.forecast, type:'scatter', mode:'lines+markers', line:{color:'#ff9800'}}], {title:'Page '+forecastPage}));
    };
    var forecast = {{ forecast|tojson }};
    Plotly.newPlot('forecast-chart', [{x:Array.from({length:30},(_,i)=>'Day '+(i+1)), y:forecast, type:'scatter', mode:'lines+markers', line:{color:'#ff9800'}}], {title:'Predicted Yield'});
    Plotly.newPlot('price-chart', [{x:['Corn','Wheat','Soybeans'], y:[4.82,5.31,12.15], type:'bar', marker:{color:'#01B763'}}], {title:'Market Prices'});
    Plotly.newPlot('price-prediction', [{x:['Week1','Week2','Week3','Week4'], y:[4.85,4.92,5.01,5.10], type:'scatter', line:{color:'#01B763'}}], {title:'Corn Price Forecast'});
    Plotly.newPlot('weather-impact', [{x:[20,22,24,26,28,30], y:[1100,1200,1300,1250,1150,1000], mode:'markers', type:'scatter', marker:{color:'#E74C3C'}}], {title:'Temp vs Yield'});
    document.getElementById('next-page-btn').addEventListener('click', function(){forecastPage++; window.refreshPageData();});
</script>
'''

MAINTENANCE_PAGE = '''
<h1 style="color:#005B31;">🔧 Predictive Maintenance</h1>
<div class="grid-2">
    <div class="card"><h3>Equipment Hours</h3><div id="hours-chart"></div></div>
    <div class="card"><h3>Maintenance Alerts</h3><div id="maintenance-alerts" class="maintenance-card"></div></div>
</div>
<script>
    fetch('/api/maintenance-data').then(res=>res.json()).then(data => {
        Plotly.newPlot('hours-chart', [{x:data.fields, y:data.hours, type:'bar', marker:{color:['#01B763','#ffb74d','#E74C3C','#01B763']}}], {title:'Equipment Hours'});
        let alerts = '';
        data.maintenance.forEach((m,i) => alerts += `<div style="padding:0.5rem; background:${m.status=='Due soon'?'#ffebee':'#e8f5e9'}; border-radius:8px; margin:0.5rem 0;">${data.fields[i]}: ${m.hours} hrs - ${m.status}</div>`);
        document.getElementById('maintenance-alerts').innerHTML = alerts;
    });
</script>
'''

CARBON_PAGE = '''
<h1 style="color:#005B31;">🌍 Carbon Footprint</h1>
<div class="grid-2">
    <div class="card"><h3>Emissions by Field</h3><div id="co2-chart"></div></div>
    <div class="card"><h3>Offset Progress</h3><div id="offset-gauge"></div></div>
</div>
<script>
    fetch('/api/carbon-data').then(res=>res.json()).then(data => {
        Plotly.newPlot('co2-chart', [{x:data.fields, y:data.emissions, type:'bar', marker:{color:'#01B763'}}], {title:'kg CO₂/ha'});
        Plotly.newPlot('offset-gauge', [{type:'indicator', mode:'gauge+number', value:data.offset, title:{text:'Offset %'}, gauge:{axis:{range:[0,100]}, bar:{color:'#01B763'}}}], {height:250});
    });
</script>
'''

ENERGY_PAGE = '''
<div class="welcome-header">
    <h1>SG-1980 Energy Monitor</h1>
    <div class="date-badge">{{ now.strftime('%d %B, %Y') }}</div>
</div>
<div class="grid-4">
    <div class="card"><h3>⚡ Charging Time</h3><div class="kpi-value">4hr 32min</div></div>
    <div class="card"><h3>🔋 Battery Capacity</h3><div class="kpi-value">2000 kWh</div></div>
    <div class="card"><h3>📈 Efficiency Trend</h3><div class="kpi-value">90%</div></div>
    <div class="card"><h3>⚙️ Energy Performance</h3><div class="kpi-value">350W</div></div>
</div>
<div class="grid-2">
    <div class="card"><h3>📊 Monthly Energy Generation</h3><div id="energy-chart"></div></div>
    <div class="card"><h3>☀️ Weather today</h3><div style="font-size: 4rem; text-align: center;">65°</div><div style="display: flex; justify-content: space-around; margin-top:1rem;"><div><span class="stat-number">4500</span>/5000<br>Active Panels</div><div><span class="stat-number">500</span><br>Faulty Panels</div></div></div>
</div>
<div class="grid-2">
    <div class="card"><h3>📉 Performance Trend</h3><div id="performance-chart"></div></div>
    <div class="card"><h3>🏡 My Farm</h3><div style="display: flex; justify-content: space-between;"><div>Total Capacity<br><span class="stat-number">10MW</span></div><div>Faulty Panels<br><span class="stat-number">500</span></div></div><div style="margin-top:2rem;"><div>Daily <span class="stat-number">$14</span></div><div>Every Generation <span class="stat-number">$55</span></div></div></div>
</div>
<script>
    var months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    var energyData = [{x: months, y: [65,70,80,85,90,95,98,96,88,78,68,60], type:'bar', marker:{color:'#01B763'}}];
    Plotly.newPlot('energy-chart', energyData, {title:'Energy Generation (kWh)'});
    var perfData = [{x:[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], y:[80,75,82,78,85,88,84,90,87,92,89,94,91,95,93], type:'scatter', mode:'lines+markers', line:{color:'#E74C3C'}}];
    Plotly.newPlot('performance-chart', perfData, {title:'Performance % over last 15 days'});
</script>
'''

SETTINGS_PAGE = '''
<h1 style="color:#005B31;">⚙️ Settings</h1>
<div class="card" style="padding:2rem;">
    <h3>Thresholds</h3>
    <div><label>Soil Moisture Min: <span id="soil-value">60%</span></label><input type="range" id="soil-min" min="0" max="100" value="60" style="width:100%;"></div>
    <div><label>Temp Max: <span id="temp-value">35°C</span></label><input type="range" id="temp-max" min="0" max="50" value="35" style="width:100%;"></div>
    <h3>Notifications</h3>
    <div><label><input type="checkbox" id="email" checked> Email</label><br><label><input type="checkbox" id="sms"> SMS</label><br><label><input type="checkbox" id="push"> Push</label></div>
    <h3>User Role</h3>
    <select id="role"><option>Admin</option><option>Viewer</option></select>
    <button id="save-settings" class="btn btn-primary" style="margin-top:1rem;">Save</button>
</div>
<script>
    document.getElementById('soil-min').addEventListener('input', function(){document.getElementById('soil-value').textContent=this.value+'%';});
    document.getElementById('temp-max').addEventListener('input', function(){document.getElementById('temp-value').textContent=this.value+'°C';});
    document.getElementById('save-settings').addEventListener('click', function(){
        fetch('/api/save-settings', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({
            soil_min:document.getElementById('soil-min').value,
            temp_max:document.getElementById('temp-max').value,
            email:document.getElementById('email').checked,
            sms:document.getElementById('sms').checked,
            push:document.getElementById('push').checked,
            role:document.getElementById('role').value
        })}).then(res=>res.json()).then(d=>alert(d.message));
    });
</script>
'''

# -------------------- Routes --------------------
@app.route('/')
def dashboard():
    kpi = get_kpi_data()
    ai_message = get_ai_recommendation()
    globals = inject_globals()
    page_html = Template(DASHBOARD_PAGE).render(kpi=kpi, ai_message=ai_message, **globals)
    return render_template_string(BASE_TEMPLATE, page_content=page_html)

@app.route('/analytics')
def analytics():
    globals = inject_globals()
    page_html = Template(ANALYTICS_PAGE).render(**globals)
    return render_template_string(BASE_TEMPLATE, page_content=page_html)

@app.route('/crop-health')
def crop_health():
    globals = inject_globals()
    page_html = Template(CROP_HEALTH_PAGE).render(**globals)
    return render_template_string(BASE_TEMPLATE, page_content=page_html)

@app.route('/irrigation')
def irrigation():
    globals = inject_globals()
    page_html = Template(IRRIGATION_PAGE).render(**globals)
    return render_template_string(BASE_TEMPLATE, page_content=page_html)

@app.route('/forecasting')
def forecasting():
    forecast = forecast_yield()
    globals = inject_globals()
    page_html = Template(FORECASTING_PAGE).render(forecast=forecast, **globals)
    return render_template_string(BASE_TEMPLATE, page_content=page_html)

@app.route('/maintenance')
def maintenance():
    globals = inject_globals()
    page_html = Template(MAINTENANCE_PAGE).render(**globals)
    return render_template_string(BASE_TEMPLATE, page_content=page_html)

@app.route('/carbon')
def carbon():
    globals = inject_globals()
    page_html = Template(CARBON_PAGE).render(**globals)
    return render_template_string(BASE_TEMPLATE, page_content=page_html)

@app.route('/energy')
def energy():
    globals = inject_globals()
    page_html = Template(ENERGY_PAGE).render(**globals)
    return render_template_string(BASE_TEMPLATE, page_content=page_html)

@app.route('/settings')
def settings():
    globals = inject_globals()
    page_html = Template(SETTINGS_PAGE).render(**globals)
    return render_template_string(BASE_TEMPLATE, page_content=page_html)

# -------------------- API Endpoints --------------------
@app.route('/api/dashboard-data')
def dashboard_data():
    farm = request.args.get('farm', 'Green Valley Farm')
    crops = request.args.get('crops', '').split(',') if request.args.get('crops') else []
    filters = {'farm': farm, 'crops': crops}
    kpi = get_kpi_data(filters)
    ai_message = get_ai_recommendation()
    df = generate_sample_data(**filters)
    fig = px.line(df, x='date', y='crop_yield', color='field', title='Yield Over Time')
    yield_chart = json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))
    satellite_data = {
        'z': [[1, 0.8, 0.7, 0.9], [0.7, 0.9, 0.8, 0.6], [0.8, 0.7, 0.9, 0.8], [0.9, 0.8, 0.7, 0.9]],
        'x': ['Field A', 'Field B', 'Field C', 'Field D'],
        'y': ['Field A', 'Field B', 'Field C', 'Field D']
    }
    return jsonify({'kpi': kpi, 'ai_message': ai_message, 'yield_chart': yield_chart, 'satellite_data': satellite_data})

@app.route('/api/clustering-data')
def clustering_data():
    farm = request.args.get('farm', 'Green Valley Farm')
    crops = request.args.get('crops', '').split(',') if request.args.get('crops') else []
    filters = {'farm': farm, 'crops': crops}
    df = generate_sample_data(**filters).groupby('field').agg({
        'temperature': 'mean', 'humidity': 'mean', 'soil_moisture': 'mean', 'crop_yield': 'mean'
    }).reset_index()
    features = df[['temperature', 'humidity', 'soil_moisture', 'crop_yield']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled)
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/anomaly-chart-data')
def anomaly_chart_data():
    fields = ['Field A', 'Field B', 'Field C', 'Field D']
    scores = np.random.uniform(-0.5, 0.5, 4).tolist()
    fig = go.Figure(data=[go.Bar(x=fields, y=scores, marker_color=['red' if s>0.2 else 'green' for s in scores])])
    fig.update_layout(title='Anomaly Scores')
    return jsonify(json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)))

@app.route('/api/ndvi-data')
def ndvi_data():
    return jsonify({'fields': ['Field A','Field B','Field C','Field D'], 'ndvi': [0.82,0.75,0.68,0.79], 'colors': ['#01B763','#ffb74d','#E74C3C','#66bb6a']})

@app.route('/api/radar-data')
def radar_data():
    return jsonify({'metrics': ['NDVI','Pest','Disease','Water','Nutrient'], 'values': [0.8,0.6,0.3,0.4,0.7]})

@app.route('/api/water-usage')
def water_usage():
    return jsonify({'days': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], 'usage': [120,135,110,145,130,155,140]})

@app.route('/api/gauge-data')
def gauge_data():
    return jsonify({'soil_moisture': np.random.randint(60,85)})

@app.route('/api/weather-forecast')
def weather_forecast():
    return jsonify({'days': ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'], 'temp': [23,25,22,21,24,26,27]})

@app.route('/api/forecast-data')
def forecast_data():
    forecast = forecast_yield()
    page = int(request.args.get('page', 1))
    if page > 1:
        forecast = [f * (1 + (page-1)*0.05) for f in forecast]
    days = ['Day '+str(i+1+(page-1)*30) for i in range(30)]
    return jsonify({'days': days, 'forecast': forecast})

@app.route('/api/price-prediction')
def price_prediction():
    return jsonify({'dates': ['Week1','Week2','Week3','Week4'], 'prices': [4.85,4.92,5.01,5.10]})

@app.route('/api/weather-impact')
def weather_impact():
    return jsonify({'temp': [20,22,24,26,28,30], 'yield': [1100,1200,1300,1250,1150,1000]})

@app.route('/api/chart-data')
def chart_data():
    df = generate_sample_data()
    fig = px.line(df, x='date', y='crop_yield', color='field', title='Yield Over Time')
    return jsonify(json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)))

@app.route('/api/sensor-data')
def sensor_data():
    return jsonify({
        'soil_moisture': round(np.random.uniform(60,85),1),
        'temperature': round(np.random.uniform(18,32),1),
        'humidity': round(np.random.uniform(50,80),1),
        'light': np.random.randint(800,1000)
    })

@app.route('/api/anomalies')
def anomalies():
    return jsonify({'anomalies': detect_anomalies()})

@app.route('/api/maintenance-data')
def maintenance_data():
    data = predict_maintenance()
    fields = list(data.keys())
    hours = [data[f][0] for f in fields]
    status = [data[f][1] for f in fields]
    return jsonify({'fields': fields, 'hours': hours, 'maintenance': [{'hours':h, 'status':s} for h,s in zip(hours,status)]})

@app.route('/api/carbon-data')
def carbon_data():
    return jsonify({
        'fields': ['Field A','Field B','Field C','Field D'],
        'emissions': [round(np.random.uniform(15,30),1) for _ in range(4)],
        'offset': np.random.randint(40,80)
    })

@app.route('/api/activate-valves', methods=['POST'])
def activate_valves():
    return jsonify({'message': 'All valves activated successfully!'})

@app.route('/api/toggle-valve', methods=['POST'])
def toggle_valve():
    data = request.json
    field = data.get('field')
    return jsonify({'message': f'Valve for Field {field} toggled.'})

@app.route('/api/save-settings', methods=['POST'])
def save_settings():
    settings = request.json
    print("Settings saved:", settings)
    return jsonify({'message': 'Settings saved successfully!'})

if __name__ == '__main__':
    print(f"Using {'REAL' if REAL_DATA_AVAILABLE else 'SYNTHETIC'} data.")
    app.run(debug=True, host='127.0.0.1', port=5000)
