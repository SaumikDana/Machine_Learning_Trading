## Directory Structure

```
Machine_Learning_Trading/
├── .ipynb_checkpoints/                  # Checkpoint files for Jupyter notebooks
│   ├── Various Jupyter notebook checkpoints (*.ipynb)
├── alpha_factors/                       # Notebooks related to alpha factors in trading strategies
│   ├── .ipynb_checkpoints/              # Checkpoints for notebooks in alpha_factors
│   │   ├── feature_engineering-checkpoint.ipynb
│   │   ├── how_to_use_talib-checkpoint.ipynb
│   │   └── kalman_filter_and_wavelets-checkpoint.ipynb
│   ├── feature_engineering.ipynb        # Notebook on feature engineering techniques
│   ├── how_to_use_talib.ipynb           # Tutorial on using TA-Lib for technical analysis
│   └── kalman_filter_and_wavelets.ipynb # Exploring Kalman filters and wavelets
├── data/                                # Directory for data and data-related scripts
│   ├── .ipynb_checkpoints/              # Checkpoints for notebooks in data
│   │   ├── create_data-checkpoint.ipynb
│   │   └── earnings_release-checkpoint.ipynb
│   ├── __pycache__/                     # Compiled Python files for faster loading
│   │   ├── analyze_stock.cpython-311.pyc
│   │   ├── implied_volatility.cpython-311.pyc
│   │   ├── scrape_url.cpython-311.pyc
│   │   ├── stock_tracker.cpython-311.pyc
│   │   └── utils.cpython-311.pyc
│   ├── analyze_stock.py                 # Script for stock analysis
│   ├── create_data.ipynb                # Notebook for data creation
│   ├── earnings_release.ipynb           # Analysis of earnings releases
│   ├── futures.ipynb                    # Notebook discussing futures
│   ├── kc_house_data.csv                # Dataset for house prices
│   ├── message_types.csv                # CSV file, related to messaging or transaction types
│   ├── message_types.xlsx               # Excel version of message_types
│   ├── scrape_url.py                    # Script for scraping URLs
│   ├── specific_stock_analysis.ipynb    # Notebook for analyzing specific stocks
│   ├── us_equities_meta_data.csv        # Metadata for US equities
│   ├── utils.py                         # Utility functions
│   └── wiki_stocks.csv                  # Dataset related to stocks
├── decision trees/                      # Notebooks and results related to decision tree models
│   ├── .ipynb_checkpoints/              # Checkpoints for notebooks in decision trees
│   │   ├── bagged_decision_trees-checkpoint.ipynb
│   │   ├── data_prep_decision_trees_random_forests-checkpoint.ipynb
│   │   └── decision_trees-checkpoint.ipynb
│   ├── bagged_decision_trees.ipynb      # Discussing bagged decision trees
│   ├── data_prep_decision_trees_random_forests.ipynb # Preparing data for decision trees and random forests
│   ├── decision_trees.ipynb             # Notebook on decision trees
│   └── results/                         # Results from decision tree models
│       └── decision_trees/
│           ├── clf_tree_t2.dot          # Dot file for a classifier tree
│           └── reg_tree_t2.dot          # Dot file for a regression tree
├── deep_learning/                       # Notebooks related to deep learning models
│   ├── build_and_train_feedforward_nn.ipynb # Building and training feedforward neural networks
│   └── how_to_use_pytorch.ipynb         # Guide on using PyTorch
├── linear_models/                       # Notebooks on linear models in trading
│   ├── fama_macbeth.ipynb               # Notebook on Fama-MacBeth regression
│   ├── linear_regression.ipynb          # Discussing linear regression
│   ├── predicting_stock_returns_with_linear_regression.ipynb # Predicting stock returns
│   └── prepping_data.ipynb              # Preparing data for linear models
├── machine_learning_process/            # Discussing the overall machine learning process
│   ├── bias_variance.ipynb              # Exploring the bias-variance tradeoff
│   └── ml_workflow.ipynb                # Notebook on the machine learning workflow
├── results/                             # General results and visualizations
│   ├── boundary.png                     # Image of a decision boundary
│   ├── decision_trees/                  # Results from decision tree models
│   │   └── reg_tree_t2.dot              # Dot file for a regression tree
│   ├── ffnn_data.png                    # Image related to feedforward neural network data
│   ├── ffnn_loss.png                    # Image of a feedforward neural network loss graph
│   ├── projection3d.png                 # 3D projection image
│   └── surface.png                      # Image of a surface plot
├── time_series_models/                  # Notebooks on time series models
│   ├── arch_garch_models.ipynb          # Discussing ARCH/GARCH models
│   ├── arima_models.ipynb               # Notebook on ARIMA models
│   ├── tsa_stationarity.ipynb           # Discussing time series stationarity
│   └── vector_autoregressive_model.ipynb # Exploring vector autoregressive models
├── unsupervised_learning/               # Notebooks on unsupervised learning techniques
│   ├── curse_dimensionality.ipynb       # Discussing the curse of dimensionality
│   ├── pca_and_risk_factor_models.ipynb # PCA and risk factor models
│   ├── pca_key_ideas.ipynb              # Key ideas behind PCA
│   └── the_math_behind_pca.ipynb        # Mathematical concepts behind PCA
└── README.md                            # Overview or instructions for the repository
```

## /data/

<img width="753" alt="strategy" src="https://github.com/SaumikDana/Machine_Learning_Trading/assets/9474631/3933ff9f-4370-40f2-ab62-dc8a73cbf945">


