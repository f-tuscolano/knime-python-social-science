# Statistics and Social Science Extension for KNIME

This repository contains the code for the Statistics and Social Science Extension for the KNIME Analytics Platform. This extension provides nodes for multivariate statistical analysis, time series modeling, and data visualizations, enabling users to perform in-depth explorations of structured data.

The extension is curated and maintained by Francesco Tuscolano (KNIME), Prof. Daniele Tonini, and Pietro Maran (Bocconi University, Milan).

The project's goal is to integrate advanced statistical methodologies within KNIME by leveraging bundled Python packages and transforming them into native KNIME nodes.

## Current Nodes

* Auto-SARIMA Learner: Automatically finds the optimal parameters for and trains a Seasonal AutoRegressive Integrated Moving Average (SARIMA) model on a given time series, returning also the in-sample predictions, residuals, and model statistics. This model is the SARIMAX class from the statsmodels library. 
* Auto-SARIMA Predictor: Generates future forecasts using a pre-trained SARIMA model, output of the node Auto-SARIMA Learner.
* Correspondence Analysis Learner: Performs Correspondence Analysis (CA) or Multiple Correspondence Analysis (MCA) on categorical data to explore associations between them.
* Factor Analyzer: Trains a factor analysis model using Principal Component Analysis (PCA) to reduce the dimensionality of numerical data. Supports optional component rotation (for improved interpretability) using rotation functions from the factor_analyzer Python package, and incremental (batch-wise) learning for handling large datasets efficiently. Outputs include principal components, explained variance, loadings, and a reusable PCA model for deployment. (Rotation functions available: varimax, promax, oblimin, and quartimax.)
* Factor Scorer: Applies a trained PCA model (from the Factor Analyzer) to new datasets, projecting data onto the learned principal components. The scorer is fully compatible with models trained using rotation or incremental PCA, ensuring consistent and reproducible dimensionality reduction across production workflows.

## Package Organization

* **`icons/`**: Directory containing visual assets and icon images for the extension nodes, including specialized icons for time series analysis, correspondence analysis, and PCA components.
* **`config.yml`**: Configuration file specifying the path to the extension source code directory. This file works in conjunction with `knime.yml` to define the extension structure and dependencies.
* **`knime.yml`**: YAML configuration file containing extension metadata, including extension identification, module paths, and KNIME integration specifications.
* **`pixi.toml` & `pixi.lock`**: Pixi package manager configuration files defining Python dependencies, environment setup, and reproducible package management for the extension.
* **`src/social_science_ext.py`**: Main extension module that registers and initializes all nodes within the Social Science Extension for KNIME Analytics Platform.
* **`src/nodes/`**: Core implementation directory containing individual node modules:
  - `arima_learner.py`: SARIMA/SARIMAX time series model fitting node with automatic parameter optimization
  - `arima_predictor.py`: Time series forecasting predictor node for applying trained ARIMA models
  - `correspondence_analysis.py`: Correspondence Analysis (CA) and Multiple Correspondence Analysis (MCA) implementation for categorical data analysis
  - `pca.py`: Principal Component Analysis learner with multiple rotation methods (Varimax, Promax, Quartimax) using factor_analyzer
  - `pca_scorer.py`: PCA predictor node for applying trained PCA models to new data with proper score transformation
* **`src/util/`**: Utility module directory containing helper functions and shared code components:
  - `utils.py`: Collection of utility functions adapted from Harvard's Spatial Data Lab, providing common data processing and validation methods

This extension provides a comprehensive suite of statistical analysis tools specifically designed for social science research within the KNIME Analytics Platform, supporting both time series analysis and multivariate statistical methods for categorical and continuous data.



