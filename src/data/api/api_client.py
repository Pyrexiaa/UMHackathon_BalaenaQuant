import requests
from dotenv import load_dotenv

from src.data.api.client_config import ClientConfig

load_dotenv()


class APIClient:
    def __init__(self, datasource_key: str):
        self.client_config = ClientConfig(datasource_key)

    def get(self, endpoint: str):
        url = f"{self.client_config.get_base_url()}/{endpoint.lstrip('/')}"
        headers = {"X-API-Key": self.client_config.get_api_key()}
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"API request failed: {e}")
            return {}
