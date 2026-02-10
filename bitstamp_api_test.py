import requests
import time
import hmac
import hashlib

API_KEY = 'qJFV9q7knofXxNRyZexiZsjO0KoZm7GG'
API_SECRET = '2wNvNcxz7PLBog3Ah7G5T1W69i5CLpwC'
CUSTOMER_ID = 'bcdu2450'

def create_signature(nonce):
    message = f"{nonce}{CUSTOMER_ID}{API_KEY}"
    signature = hmac.new(API_SECRET.encode(), msg=message.encode(), digestmod=hashlib.sha256).hexdigest()
    return signature.upper()

nonce = str(int(time.time() * 1000))
signature = create_signature(nonce)
url = "https://www.bitstamp.net/api/v2/balance/"
headers = {"Content-Type": "application/x-www-form-urlencoded"}
payload = {"key": API_KEY, "signature": signature, "nonce": nonce}

response = requests.post(url, data=payload, headers=headers)
print(response.json())
