import os
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from src.data.data_handler import DataHandler
from src.data.data_source import feature_topic_dict
import pandas as pd
import numpy as np
# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI()

# Data processing function
def process_data(source_key: str, endpoint_key: str):
    """
    Loads, cleans, and enriches data using the DataHandler, and returns the processed data.
    """
    print(f"Starting data pipeline for {endpoint_key.upper()} from source '{source_key}'...")

    # Create DataHandler instance
    handler = DataHandler(source_key=source_key, endpoint_key=endpoint_key)

    # Load, clean, and enrich the data
    handler.load_data()
    # handler.clean_data()
    # handler.add_features()

    # Retrieve processed data
    df = handler.get_data()

    # Optional: Save the processed data to CSV
    output_path = f"output/{source_key}_{endpoint_key}_processed.csv"
    os.makedirs("output", exist_ok=True)
    df.to_csv(output_path)
    print(f"Processed data saved to {output_path}")

    return df.head().to_dict(orient="records")  # Return a preview as a dictionary

async def extract_all_features():
    print("Extracting all features...")
    os.makedirs("output", exist_ok=True)
    csv_paths = []

    handler = DataHandler(source_key="cryptoquant", endpoint_key="reserve")

    for key in feature_topic_dict.keys():
        print(f"Extracting: {key}")
        await handler.extract_features(feature_topic_dict[key])

        df = handler.get_data()

        if df is None or df.empty:
            print(f"⚠️ No data returned for {key}, skipping...")
            continue

        output_path = f"output/{key}.csv"
        df.to_csv(output_path, index=False)
        csv_paths.append(output_path)
        print(f"✅ Saved {key} to {output_path}")

        handler.reset_data()

    if not csv_paths:
        raise ValueError("No valid data was extracted. All DataFrames were empty.")

    # Join all CSVs by 'time'
    print("📎 Joining all CSVs on 'time'...")
    merged_df = None
    for path in csv_paths:
        try:
            df = pd.read_csv(path, parse_dates=["datetime"])
            df.set_index("datetime", inplace=True)

            if merged_df is None:
                merged_df = df
            else:
                suffix = os.path.basename(path).replace(".csv", "")
                merged_df = merged_df.join(df, how="inner", rsuffix=f"_{suffix}")
        except Exception as e:
            print(f"❌ Failed to process {path}: {e}")

    if merged_df is None:
        raise ValueError("No DataFrames were successfully joined.")

    merged_df.reset_index(inplace=True)

    merged_output_path = "output/merged_features.csv"
    merged_df.to_csv(merged_output_path, index=False)
    print(f"✅ Joined CSV saved to {merged_output_path}")
    
    merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    merged_df = merged_df.where(pd.notnull(merged_df), None)
    
    merged_df["datetime"] = merged_df["datetime"].astype(str)
    return merged_df.head().to_dict(orient="records")


# Define FastAPI route to process data
@app.get("/process-data")
async def process_data_endpoint(source_key: str = "", endpoint_key: str = ""):
    """
    FastAPI route to trigger the data processing.
    Parameters:
        source_key (str): The source key (e.g., "cybotrade").
        symbol (str): The symbol to fetch data for (e.g., "btc").
        timeframe (str): The timeframe for the data (default is "1d").
    """
    try:
        data = process_data(source_key,endpoint_key)
        return JSONResponse(content={"message": "Data processed successfully", "data": data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing data: {str(e)}"})

@app.get("/extract-features")
async def extract_all_features_endpoint():
    try:
        data = await extract_all_features()
        return JSONResponse(content={"message": "Extracted all features successfully", "data": data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error extracting features: {str(e)}"})
    
# FastAPI root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}
