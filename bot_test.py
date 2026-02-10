import requests

BOT_TOKEN = '8177096945:AAHTg5nxVTA6hcPkAL4MLAAHPsZ1at7Ywmw'
URL = f'https://api.telegram.org/bot{BOT_TOKEN}/getUpdates'

response = requests.get(URL)
data = response.json()

# Pretty-print the updates
import json
print(json.dumps(data, indent=2))
