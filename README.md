## Directory Structure

```
Machine_Learning_Trading/
│
├── .ipynb_checkpoints/ (Various Jupyter notebook checkpoints)
│
├── alpha_factors/
│   ├── .ipynb_checkpoints/ (Jupyter notebook checkpoints)
│   ├── feature_engineering.ipynb (Notebook on feature engineering in trading)
│   ├── how_to_use_talib.ipynb (Notebook on using TA-Lib for technical analysis)
│   └── kalman_filter_and_wavelets.ipynb (Notebook on Kalman filters and wavelets in trading)
│
├── data/
│   ├── .ipynb_checkpoints/
│   │   ├── create_data-checkpoint.ipynb (Checkpoint for data creation notebook)
│   │   └── earnings_release-checkpoint.ipynb (Checkpoint for earnings release analysis notebook)
│   ├── __pycache__/ (Compiled Python files)
│   ├── Brownian Motion for Quants _ Stochastic Calculus.ipynb (Notebook on Brownian motion and stochastic calculus)
│   ├── FeatureEng_StockAnalysis.ipynb (Notebook on feature engineering for stock analysis)
│   ├── ^spx_d.csv (CSV file, possibly S&P 500 daily data)
│   ├── analyze_stock.py (Python script for stock analysis)
│   ├── black_scholes.py (Python script for Black-Scholes model)
│   ├── create_data.ipynb (Notebook for data creation)
│   ├── earnings_release.ipynb (Notebook analyzing earnings releases)
│   ├── futures.ipynb (Notebook on futures trading)
│   ├── kc_house_data.csv (CSV file, possibly King County house data)
│   ├── markovian_v_non-markovian.ipynb (Notebook on Markovian vs non-Markovian processes)
│   ├── message_types.csv (CSV file, possibly related to messaging or data types)
│   ├── message_types.xlsx (Excel file, similar to message_types.csv)
│   ├── scrape_url.py (Python script for URL scraping)
│   ├── specific_stock_analysis.ipynb (Notebook for analysis of specific stocks)
│   ├── us_equities_meta_data.csv (CSV file with metadata on US equities)
│   ├── utils.py (Python utility script)
│   ├── wiener_process.ipynb (Notebook on Wiener processes)
│   └── wiki_stocks.csv (CSV file, possibly related to stock data from Wikipedia)
│
├── decision trees/
│   ├── .ipynb_checkpoints/ (Jupyter notebook checkpoints)
│   ├── bagged_decision_trees.ipynb (Notebook on bagged decision trees)
│   ├── data_prep_decision_trees_random_forests.ipynb (Notebook on data preparation for decision trees and random forests)
│   ├── decision_trees.ipynb (Notebook on decision trees)
│   └── results/
│       └── decision_trees/
│           ├── clf_tree_t2.dot (Graphviz dot file for a classification tree)
│           └── reg_tree_t2.dot (Graphviz dot file for a regression tree)
│
├── deep_learning/
│   ├── build_and_train_feedforward_nn.ipynb (Notebook on building and training feedforward neural networks)
│   └── how_to_use_pytorch.ipynb (Notebook on using PyTorch for deep learning)
│
├── linear_models/
│   ├── fama_macbeth.ipynb (Notebook on Fama-MacBeth regression in finance)
│   ├── linear_regression.ipynb (Notebook on linear regression models)
│   ├── predicting_stock_returns_with_linear_regression.ipynb (Notebook on predicting stock returns using linear regression)
│   └── prepping_data.ipynb (Notebook on data preparation for linear models)
│
├── machine_learning_process/
│   ├── bias_variance.ipynb (Notebook on bias-variance tradeoff in machine learning)
│   └── ml_workflow.ipynb (Notebook on machine learning workflow)
│
├── time_series_models/
│   ├── arch_garch_models.ipynb (Notebook on ARCH and GARCH models in time series)
│   ├── arima_models.ipynb (Notebook on ARIMA models in time series)
│   ├── tsa_stationarity.ipynb (Notebook on time series stationarity)
│   └── vector_autoregressive_model.ipynb (Notebook on vector autoregressive models)
│
├── unsupervised_learning/
│   ├── curse_dimensionality.ipynb (Notebook on the curse of dimensionality)
│   ├── pca_and_risk_factor_models.ipynb (Notebook on PCA and risk factor models)
│   ├── pca_key_ideas.ipynb (Notebook on key ideas of PCA)
│   └── the_math_behind_pca.ipynb (Notebook on the mathematics behind PCA)
│
├── results/
│   ├── boundary.png (Image file, possibly a plot or graph)
│   ├── decision_trees/
│   │   └── reg_tree_t2.dot (Graphviz dot file for a regression tree)
│   ├── ffnn_data.png (Image file, possibly related to feedforward neural network data)
│   ├── ffnn_loss.png (Image file, possibly a loss graph for a neural network)
│   ├── projection3d.png (3D projection image file)
│   └── surface.png (Image file, possibly a surface plot)
│
└── README.md (Readme file with repository overview)
```

## /data/

![strategy](https://github.com/SaumikDana/Machine_Learning_Trading/assets/9474631/2a1de833-3538-4193-9892-df2ad5f29bfd)


## Concomitant Focus Areas

- Practical Trading Algorithms: Linear regression, decision trees, and neural networks, tailored to predict market movements and analyze financial data.
- Time Series Analysis: Specialized notebooks on time series models such as ARIMA and GARCH provide insights into handling and forecasting sequential data, crucial for market trend analysis.
- Unsupervised Learning Techniques: Exploration of complex financial datasets with unsupervised learning methods like PCA, uncovering hidden structures and reducing dimensionality for better insights.
- Alpha Factor Research: Investigation of alpha factors and feature engineering to create predictive signals and enhance trading strategies.
- Real-World Data: Engagement with real-world datasets including stock prices, earnings releases, and economic indicators to practice and apply learned concepts.
