from abc import ABC, abstractmethod
from datetime import datetime
import os
from typing import Dict
import numpy as np
import pandas as pd

from data.api.api_client import APIClient
from data.constant.data_source import data_source

class BaseLoader(ABC):
    """
    BaseLoader is an abstract class that serves as a template for loading data from a specific data source. 
    """
    
    def __init__(self, datasource_key: str):
        """
        Initialize the BaseLoader class.

        :param datasource_key: Key to access the specific data source from configuration.
        """
        self.datasource_key = datasource_key
        self.datasource = data_source[datasource_key]  # Load data source configuration
        self.api_client = APIClient(datasource_key)  # Initialize the API client
        self.data = pd.DataFrame()  # DataFrame to store loaded data
        self.dataframes: Dict[str, pd.DataFrame] = {}  # Dictionary to store multiple DataFrames by metric name
    
    @abstractmethod
    async def load_data(self) -> None:
        """
        Abstract method to be implemented by subclasses to load data.
        """
        pass
    
    def clean_data(self) -> None:
        """
        Cleans the loaded data by removing duplicates and filling missing values.
        """
        self.data = self.data.drop_duplicates()  # Drop duplicate rows
        self.data = self.data.fillna(method='ffill')  # Forward fill missing values

    def get_data(self) -> pd.DataFrame:
        """
        Retrieves the cleaned data.

        :return: The cleaned DataFrame.
        """
        return self.data

    def reset_data(self):
        """
        Resets the data to an empty DataFrame.
        This can be used to clear any loaded or processed data.
        """
        self.data = pd.DataFrame()  # Reset data to empty DataFrame
        print(self.data)
        print("Data reset to an empty DataFrame.")
    
    def save_data_to_csv(self, datasource: str, metrics: str) -> None:
        """
        Saves the cleaned data to a CSV file.

        :param datasource: The name of the data source.
        :param metrics: The metric name for the file naming.
        """
        output_path = f"output/{datasource}_{metrics}_processed.csv"  # File path for the output CSV
        os.makedirs("output", exist_ok=True)  # Create output directory if it doesn't exist
        self.get_data().to_csv(output_path)  # Save the cleaned data to CSV
        print(f"Processed data saved to {output_path}")
    
    def merge_csv(self, datasource: str, metrics: list[str]) -> None:
        """
        Merges CSV files corresponding to multiple metrics into a single DataFrame.

        :param datasource: The name of the data source.
        :param metrics: List of metric names to be merged.
        :return: A DataFrame containing the merged data.
        """
        # Generate the file paths for the metrics
        csv_paths = [f"output/{datasource}_{metric}_processed.csv" for metric in metrics]
        print(f"CSV paths: {csv_paths}")
        
        merged_df = None  # Initialize merged DataFrame

        for df_name in self.dataframes.keys():
            try:
                df = self.dataframes[df_name]  # Get DataFrame by metric name
                df.set_index("datetime", inplace=True)  # Set 'datetime' as the index for merging

                # Merge the DataFrames
                if merged_df is None:
                    merged_df = df
                else:
                    suffix = os.path.basename(df_name)  # Generate a suffix based on the DataFrame name
                    merged_df = merged_df.join(df, how="inner", rsuffix=f"_{suffix}")  # Perform an inner join

            except Exception as e:
                print(f"Failed to process {df_name}: {e}")  # Error handling for failed processing

        # Check if the merge was successful
        if merged_df is None:
            raise ValueError("No DataFrames were successfully joined.")  # Raise an error if no DataFrames were merged

        merged_df.reset_index(inplace=True)  # Reset the index after merging

        # Generate the output path for the merged file
        merged_output_path = f"output/{datasource}_{datetime.now()}_merged.csv"
        merged_df.to_csv(merged_output_path, index=False)  # Save the merged DataFrame to CSV
        print(f"Joined CSV saved to {merged_output_path}")
        
        # Handle infinite values and replace with NaN
        merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        merged_df = merged_df.where(pd.notnull(merged_df), None)  # Replace NaN with None
        
        # Ensure 'datetime' is of type string
        merged_df["datetime"] = merged_df["datetime"].astype(str)

        # Return a sample of the merged data as a dictionary
        return merged_df.head().to_dict(orient="records")
    
    @abstractmethod
    async def run(self) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses to run the data loading and processing pipeline.

        :return: The final processed DataFrame.
        """
        pass
