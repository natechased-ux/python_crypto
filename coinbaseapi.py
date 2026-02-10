import requests
import time
import hmac
import hashlib
import json
import jwt
from cryptography.hazmat.primitives import serialization
import secrets



# Coinbase API Configuration
API_KEY = "organizations/f38a576b-66ef-4a38-ad08-53e2bc462b62/apiKeys/2d801bc5-2863-4859-ab09-0569e3ae33a8"
API_SECRET = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIAXLO1waZap0EOI812WwiUs3NgfkNdtSrnezGg0zHT8zoAoGCCqGSM49\nAwEHoUQDQgAEf96IUsW2Bea9pGIf9kyTPbOThaOBCX/1qtrvvIkv3kFlUr2gQ7d2\nMnNDLWQMIZT28mBdpsTi109edefN24Eg7A==\n-----END EC PRIVATE KEY-----\n"
BASE_URL = "https://api.exchange.coinbase.com"

private_key_bytes = key_secret.encode('utf-8')
private_key = serialization.load_pem_private_key(private_key_bytes, password=None)

# Function to generate the signature
def generate_signature(timestamp, method, request_path, body=""):
    message = f"{timestamp}{method.upper()}{request_path}{body}"
    hmac_key = bytes(API_SECRET, 'utf-8')
    signature = hmac.new(hmac_key, message.encode('utf-8'), hashlib.sha256)
    return signature.hexdigest()

# Function to make an authenticated request
def send_authenticated_request(method, request_path, body=None):
    timestamp = str(time.time())
    body_json = json.dumps(body) if body else ""
    signature = generate_signature(timestamp, method, request_path, body_json)

    headers = {
        "CB-ACCESS-KEY": API_KEY,
        "CB-ACCESS-SIGN": signature,
        "CB-ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }

    url = BASE_URL + request_path
    if method.upper() == "GET":
        response = requests.get(url, headers=headers)
    elif method.upper() == "POST":
        response = requests.post(url, headers=headers, data=body_json)
    elif method.upper() == "DELETE":
        response = requests.delete(url, headers=headers)
    else:
        raise ValueError("Invalid method")

    return response.json()

# Function to check API key permissions
def check_permissions():
    request_path = "/profiles"
    response = send_authenticated_request("GET", request_path)

    if isinstance(response, dict) and "message" in response:
        print(f"Error: {response['message']}")
    else:
        print("Permissions Information:")
        for profile in response:
            print(f"Profile ID: {profile['id']}")
            print(f"Name: {profile['name']}")
            print(f"Can View: {profile.get('can_view', 'Unknown')}")
            print(f"Can Trade: {profile.get('can_trade', 'Unknown')}")
            print(f"Can Transfer: {profile.get('can_transfer', 'Unknown')}")
            print("-" * 30)

# Example Usage
if __name__ == "__main__":
    try:
        check_permissions()
    except Exception as e:
        print("Error:", str(e))
