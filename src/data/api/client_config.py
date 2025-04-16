import os
from src.data.constant.data_source import data_source


class ClientConfig:
    def __init__(self, datasource_key: str):
        self.api_key = data_source[datasource_key]["api_key"] or os.getenv(f"{datasource_key.upper()}_API_KEY")
        self.base_url = (data_source[datasource_key]["base_url"] or os.getenv(f"{datasource_key.upper()}_BASE_URL")).rstrip("/")
        
        if not self.api_key:
            raise ValueError("API key not found in environment variables.")
    
    def get_api_key(self) -> str:
        return self.api_key

    def get_base_url(self) -> str:
        return self.base_url