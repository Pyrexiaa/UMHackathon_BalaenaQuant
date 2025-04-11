from setuptools import setup, find_packages

setup(
    name='backtest',
    version='0.1.0',
    description='A Python library for backtesting quantitative trading strategies.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/quantbacktest',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'xgboost>=1.5.0',
        # Add more dependencies as needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
