import os
from dotenv import load_dotenv
from backtester.data.data_handler import DataHandler
from fastapi import FastAPI
from fastapi.responses import JSONResponse

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

# FastAPI root endpoint
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI!"}
