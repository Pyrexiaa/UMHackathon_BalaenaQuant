import pandas as pd
from data.loaders.base_loader import BaseLoader

class APILoader(BaseLoader):
    """
    APILoader is responsible for loading data from an API, processing it and saving the data to CSV files if needed. 
    """

    async def load_data(
        self, 
        metrics: list[str], 
        window: str = "hour", 
        limit: int = 46211, 
        save_data: bool = True, 
        merged: bool = False
    ) -> None:
        """
        Loads data from an API for the given metrics and processes it.

        :param metrics: List of metrics to load from the API (e.g., ['metric1', 'metric2']).
        :param window: The time window for the data (default is 'hour').
        :param limit: The maximum number of data points to fetch (default is 46211).
        :param save_data: Whether to save the loaded data to CSV files (default is True).
        :param merged: Whether to merge data for multiple metrics into one dataset (default is False).
        """
        
        for metric in metrics:
            try:
                # Construct the endpoint with query parameters
                endpoint = self.datasource["endpoints"][metric]
                if "?" not in endpoint:
                    endpoint_with_query_param = f"{endpoint}?window={window}&limit={limit}"
                else:
                    endpoint_with_query_param = f"{endpoint}&window={window}&limit={limit}"
                
                # Make the API call
                response = self.api_client.get(endpoint_with_query_param)

                # Process the response and load it into a DataFrame
                self.data = pd.DataFrame(response['data']) if 'data' in response else pd.DataFrame(response)

                # Convert the 'timestamp' column to datetime and set it as the index
                if 'timestamp' in self.data.columns:
                    self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
                    self.data.set_index('timestamp', inplace=True)

                # Sort the DataFrame by index (timestamp)
                self.data.sort_index(inplace=True)

                # Store the data for the current metric
                self.dataframes[metric] = self.data
                print("Data loaded and indexed.")
                
                # Save the data to CSV if requested
                if save_data:
                    self.save_data_to_csv(self.datasource_key, metric)

            except Exception as e:
                print(f"Error loading data: {e}")
                    
        # If there are multiple metrics and merging is requested, merge them
        if len(metrics) > 1 and merged:
            self.merge_csv(self.datasource_key, metrics)

    async def run(
        self, 
        metrics: list[str], 
        window: str = "hour", 
        limit: int = 46211, 
        save_data: bool = True, 
        merged: bool = False
    ) -> pd.DataFrame:
        """
        Runs the data loading and processing pipeline.

        :param metrics: List of metrics to load from the API.
        :param window: The time window for the data.
        :param limit: The maximum number of data points to fetch.
        :param save_data: Whether to save the loaded data to CSV files.
        :param merged: Whether to merge data for multiple metrics into one dataset.
        
        :return: A DataFrame containing the processed data.
        """
        # Load the data for the provided metrics
        await self.load_data(metrics, window, limit, save_data, merged)

        # Clean the data after loading
        self.clean_data()

        # Return the cleaned data
        return self.get_data()
