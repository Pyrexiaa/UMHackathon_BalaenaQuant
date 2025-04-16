import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
import asyncio
from src.data import DataLoader
from src.features import FeaturePipeline

async def main():
    loader = DataLoader()
    df = await loader.run() # load all metrics

    pipeline = FeaturePipeline.standard_features()
    new_df = pipeline.add_features(df)

    new_df.to_csv('output_data.csv', index=False)

if __name__ == "__main__":
    asyncio.run(main())