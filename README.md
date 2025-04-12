# QuantPilot

## Project Overview
QuantPilot is a high-performance, modular framework tailored for developing, backtesting, and evaluating trading strategies. It is built to support systematic strategy research using both traditional and machine learning-based approaches. With a flexible architecture and seamless integration across modules, BalaenaQuant empowers researchers and traders to rapidly iterate, test, and deploy alpha-generating ideas with real-world constraints and metrics.

## Directory Structure
### data
The data ingestion module enables structured retrieval and preprocessing of financial time-series data. It supports multiple sources via API clients or custom topic loaders and integrates with the cybotrade-resource package. This design simplifies data handling for both historical and real-time pipelines, enabling consistent data structures across research and live environments.

### features
A modular and extensible feature engineering framework that supports:
- Technical indicators (e.g., RSI, MACD, Moving Averages)
- Statistical features (volatility, skewness, kurtosis)
- ML-based transformations and signal enhancement
- Custom feature pipelines via a unified base interface

Each feature module can be independently tested, reused, and composed into pipelines for model training or strategy generation.

### metrics
The performance evaluation module, responsible for computing:
- Return metrics (e.g., CAGR, Sharpe, Sortino)
- Risk metrics (e.g., Max Drawdown, Value-at-Risk)
- Trade-level analytics (e.g., win rate, average hold time)
- Custom metrics for ML-based evaluation (e.g., precision, recall on signals)

### models
This directory contains machine learning models for signal generation, including:
- Classification models (e.g., logistic regression, random forests)
- Time-series forecasting models (e.g., ARIMA, LSTMs)
- Online learning or reinforcement learning setups All models adhere to a standard interface for training, inference, and persistence.

### strategy
Hosts the strategy composition engine, including:
- Strategy interface definitions and lifecycle hooks
- Rule-based strategies (e.g., MA crossovers)
- ML signal-based strategies
- Hybrid strategies combining multiple signal types

### backtester
The core execution engine, comprising:
- Data handler and loader orchestration
- Signal generation and order execution simulators
- Portfolio and position management
- Event-driven or time-driven backtesting logic Built for scalability, it supports single asset or multi-asset portfolios, as well as intraday or daily granularity.

### experimental
A sandbox environment for rapid prototyping, testing new strategies, features, or integrations. This is where bleeding-edge ideas take shape before being hardened for production.


### example
Includes modular and runnable pipeline scripts that demonstrate end-to-end use cases:
- Feature generation
- Model training and evaluation
- Strategy backtesting and metric reporting Ideal for new contributors or researchers to get up to speed quickly.

---

```py
root/
    │
    backtester/
    │
    ├──api/
    ├──data/
    docs/
    │
    examples/
    │
    ├──sample_data/
    experimental/
    │
    ├──data_scraping
    ├──datasets
    ├──feature_engineering
    ├──modelling
    │
    src/
    │
    ├──data
    ├──features
    ├──metrics
    ├──models
    ├──strategy
    ├──backtester.py
    ├──config.py
    │
    output/
```



## Getting Started
### Prerequisites
- Python 3.10
- Required dependencies (can be installed via requirements.txt -> `pip install -r requirments.txt`)

## Presentation Slides
https://www.canva.com/design/DAGkV7mcnvA/eA4PIpeJ23_QIn8wIw_kfg/edit?utm_content=DAGkV7mcnvA&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton

## Framework Architecture Diagram 