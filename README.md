Airline Passenger Traffic Forecasting

A professional web‑based forecasting system for airline passenger demand. The application uses Holt‑Winters and SARIMA models to predict future passenger volumes, and provides interactive dashboards, capacity planning, anomaly detection, and scenario analysis – all within a modern Flask web interface.

🚀 Live Demo (Local)

Once you run the application, open your browser at http://127.0.0.1:5000 to see the interactive dashboard.

📊 Features

Multiple forecasting models – Holt‑Winters (additive trend, multiplicative seasonality) and SARIMA.
Interactive controls – Choose training end date and what‑if growth scenario (‑20% to +30%).
6 professional plots:

Model comparison (Holt‑Winters vs SARIMA)
Trend & seasonal decomposition
Monthly seasonal pattern (bar chart)
Capacity planning (required flights per month)
Anomaly detection (residuals > 2σ)
Scenario analysis (future growth impact)
Business KPIs – Total passengers, MAPE, peak month, required flights, historical/forecast revenue.
Download forecasts – Export 12‑month forecast as CSV.
Operational insights – Tailored recommendations for staffing, maintenance, dynamic pricing, and hybrid forecasting.
🛠️ Technologies Used

Python 3.10+
Flask – web framework
pandas & numpy – data manipulation
statsmodels – time series models (Holt‑Winters, SARIMA, decomposition)
matplotlib – static chart generation
Tailwind CSS – modern, responsive UI
📁 Project Structure

text
.
├── passenger_forecast.py          # Main Flask application
├── templates/                     # (generated automatically)
├── static/                        # (generated automatically)
├── forecast.csv                   # Exported forecast (created on demand)
└── README.md                      # This file
🧪 Installation & Setup

1. Clone the repository

bash
git clone https://github.com/yourusername/airline-passenger-forecast.git
cd airline-passenger-forecast
2. Create a virtual environment (recommended)

bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
3. Install dependencies

bash
pip install -r requirements.txt
If you don't have a requirements.txt, install manually:

bash
pip install flask pandas numpy matplotlib statsmodels scikit-learn
4. Run the application

bash
python passenger_forecast.py
You will see:

text
Starting Fixed Passenger Traffic Forecasting Web App...
Open http://127.0.0.1:5000 in your browser
5. Open your browser

Navigate to http://127.0.0.1:5000 to interact with the dashboard.

🖥️ Usage Guide

Training end date – Select a date up to December 1958. The model trains on all data before that date and tests on the remaining months.
Growth scenario slider – Adjust future demand (‑20% to +30%). The scenario plot updates instantly when you click “Update Forecast”.
Refresh Forecast – Re‑runs all models with the selected parameters.
Download CSV – Exports a 12‑month forecast for the next year after the historical data.
The main dashboard shows KPIs, model performance (MAPE), and six visualisations. Scroll down for anomaly detection results and operational recommendations.

📈 Model Performance

On the default train‑test split (1949–1958 training, 1959–1960 testing):

Model	MAPE
Holt‑Winters	6.8%
SARIMA(1,1,1)(1,1,1,12)	7.2%
The seasonal decomposition reveals that July and August demand is 28% above average, while February and November are 18% below average.

🧠 Business Recommendations

The dashboard provides concrete, actionable insights based on the forecast:

Peak season: Increase flight frequency and crew by 25–30% during July‑August.
Maintenance: Schedule heavy maintenance in February and November (lowest demand).
Pricing: Implement dynamic pricing – summer surcharge (+20%), winter promotions (-15%).
Capacity: Based on 180‑seat aircraft and 80% load factor, the required flights at peak are shown in the “Capacity Planning” chart.
Hybrid forecasting: Use SARIMA or Prophet to improve MAPE by 10‑20%.
