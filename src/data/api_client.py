import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class APIClient:
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("CYBOTRADE_API_KEY")
        if not self.api_key:
            raise ValueError("API key not found in environment variables.")

    def get(self, endpoint: str):
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        headers = {"X-API-Key": self.api_key}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return {}
