from flask import Flask, request, jsonify, render_template
import requests
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import io
import base64

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configuration
app.config['PRICE_API_KEY'] = os.getenv('PRICE_API_KEY', 'your_api_key')
app.config['CACHE_DURATION'] = 3600
app.config['METADATA_CACHE_DURATION'] = 43200

# Caches
price_cache = {}
metadata_cache = {'data': None, 'timestamp': datetime.min}

URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"

@app.route('/')
def index():
    max_date = datetime.today().strftime('%Y-%m-%d')
    return render_template('index.html', max_date=max_date)

def get_last_year_prediction(state, district, market, commodity, target_date):
    last_year_base = datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=365)
    prices, days = [], []

    # Fetch historical data
    for offset in range(-7, 8):
        test_date = (last_year_base + timedelta(days=offset)).strftime("%d/%m/%Y")
        try:
            params = {
                'api-key': app.config['PRICE_API_KEY'],
                'format': 'json',
                'limit': 100,
                'filters[state]': state,
                'filters[district]': district,
                'filters[market]': market,
                'filters[commodity]': commodity,
                'filters[arrival_date]': test_date
            }
            response = requests.get(URL, params=params, timeout=10)
            records = response.json().get("records", [])
            for rec in records:
                if rec.get("max_price"):
                    prices.append(float(rec["max_price"]) / 100)
                    days.append((datetime.strptime(rec["arrival_date"], "%d/%m/%Y") - last_year_base).days)
        except:
            continue

    if len(prices) >= 3:
        X = np.array(days).reshape(-1, 1)
        y = np.array(prices)
        model = LinearRegression().fit(X, y)
        predicted_price = round(model.predict([[0]])[0], 2)
        accuracy = model.score(X, y) * 100  # R^2 score to percentage

        # Plot
        fig, ax = plt.subplots()
        ax.scatter(days, prices, color='blue', label='Actual Data')
        ax.plot(days, model.predict(X), color='red', label='Prediction')
        ax.set_xlabel('Days from Target Date')
        ax.set_ylabel('Price (INR)')
        ax.set_title(f'Price Prediction for {commodity} in {market}')
        ax.legend()

        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        img_base64 = base64.b64encode(img.getvalue()).decode('utf-8')

        return predicted_price, accuracy, img_base64
    return None, None, None

@app.route('/api/price', methods=['GET'])
def get_price():
    data = request.args
    required_fields = ['state', 'district', 'market', 'commodity']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required parameters'}), 400

    date_param = data.get('date', '')
    cache_key = f"{data['state']}_{data['district']}_{data['market']}_{data['commodity']}_{date_param}"
    cached_data = price_cache.get(cache_key)
    if cached_data and (datetime.now() - cached_data['timestamp']).seconds < app.config['CACHE_DURATION']:
        return jsonify(cached_data['data'])

    try:
        params = {
            'api-key': app.config['PRICE_API_KEY'],
            'format': 'json',
            'limit': 100,
            'filters[state]': data['state'],
            'filters[district]': data['district'],
            'filters[market]': data['market'],
            'filters[commodity]': data['commodity']
        }

        if date_param:
            try:
                parsed_date = datetime.strptime(date_param, '%Y-%m-%d').strftime('%d/%m/%Y')
                params['filters[arrival_date]'] = parsed_date
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400

        response = requests.get(URL, params=params, timeout=10)
        api_data = response.json().get('records', [])

        if not api_data:
            return jsonify({'error': 'No price data found'}), 404

        max_prices, min_prices, units = [], [], set()
        for record in api_data:
            try:
                max_prices.append(float(record.get("max_price", 0)) / 100)
                min_prices.append(float(record.get("min_price", 0)) / 100)
                units.add(record.get("unit", "kg"))
            except (ValueError, TypeError):
                continue

        result = {
            "commodity": data['commodity'],
            "market": data['market'],
            "district": data['district'],
            "state": data['state'],
            "max_price": round(sum(max_prices) / len(max_prices), 2),
            "min_price": round(sum(min_prices) / len(min_prices), 2),
            "unit": units.pop() if units else "kg",
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": "https://api.data.gov.in"
        }

        # Predict prices and calculate model accuracy if a date is given
        if date_param:
            predicted_price, accuracy, graph = get_last_year_prediction(
                data['state'], data['district'], data['market'], data['commodity'], date_param
            )
            if predicted_price:
                result['predicted_price'] = predicted_price
                result['model_accuracy'] = round(accuracy, 2)
                result['graph'] = graph  # Attach graph to the response

        price_cache[cache_key] = {'data': result, 'timestamp': datetime.now()}
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/metadata')
def get_metadata():
    if metadata_cache['data'] and (datetime.now() - metadata_cache['timestamp']).seconds < app.config['METADATA_CACHE_DURATION']:
        return jsonify(metadata_cache['data'])

    try:
        response = requests.get(URL, params={
            'api-key': app.config['PRICE_API_KEY'],
            'format': 'json',
            'limit': 10000
        }, timeout=10)

        records = response.json().get('records', [])
        meta = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))
        commodities = set()

        for record in records:
            state = record.get("state")
            district = record.get("district")
            market = record.get("market")
            commodity = record.get("commodity")
            if state and district and market:
                meta[state][district][market].add(commodity)
                commodities.add(commodity)

        structured_data = {
            state: {
                dist: {
                    market: list(items)
                    for market, items in markets.items()
                }
                for dist, markets in dists.items()
            } for state, dists in meta.items()
        }

        final_data = {
            'structure': structured_data,
            'all_commodities': sorted(list(commodities))
        }

        metadata_cache['data'] = final_data
        metadata_cache['timestamp'] = datetime.now()
        return jsonify(final_data)

    except Exception as e:
        return jsonify({'error': f'Metadata fetch error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
