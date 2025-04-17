import asyncio

from quantpilot.data import DataLoader
from quantpilot.features import FeaturePipeline

async def main():
    loader = DataLoader()
    df = await loader.run() # load all metrics

    pipeline = FeaturePipeline.standard_features()
    new_df = pipeline.add_features(df)

    new_df.to_csv('output_data.csv', index=False)

if __name__ == "__main__":
    asyncio.run(main())