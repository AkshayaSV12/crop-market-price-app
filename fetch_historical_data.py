import requests
import pandas as pd
from datetime import datetime
from tqdm import tqdm

API_KEY = "579b464db66ec23bdd0000017008cf24063047f162ec28cf865d5342"
URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
YEAR = 2025

params = {
    'api-key': API_KEY,
    'format': 'json',
    'limit': 1000,
    'offset': 0,
    'filters[date]': '',  # we'll use this to loop through dates
}

all_data = []

# Loop over months and days
for month in tqdm(range(1, 13)):
    for day in range(1, 32):
        try:
            # Construct date string
            date_str = f"{YEAR}-{month:02d}-{day:02d}"
            params['filters[date]'] = date_str

            # Make the API request
            response = requests.get(URL, params=params, timeout=10)

            # Check the status code and print the response for debugging
            if response.status_code == 200:
                records = response.json().get('records', [])
                
                if not records:
                    print(f"No records found for {date_str}.")
                
                all_data.extend(records)
            else:
                print(f"Failed request for {date_str}: {response.status_code}")
        except Exception as e:
            print(f"Skipping {date_str} due to error: {e}")

# Check if any data was fetched
if all_data:
    # Convert to DataFrame and save
    df = pd.DataFrame(all_data)
    df.to_csv("agmarknet_prices_2024.csv", index=False)
    print(f"Saved {len(df)} records.")
else:
    print("No data was saved.")
