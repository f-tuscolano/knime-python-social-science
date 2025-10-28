# Statistics and Social Science Extension for KNIME

This repository contains the code for the Statistics and Social Science Extension for the KNIME Analytics Platform. This extension provides nodes for multivariate statistical analysis, time series modeling, and data visualizations, enabling users to perform in-depth explorations of structured data.

The extension is curated and maintained by Francesco Tuscolano (KNIME), Prof. Daniele Tonini, and Pietro Maran (Bocconi University, Milan).

The project's goal is to integrate advanced statistical methodologies within KNIME by leveraging bundled Python packages and transforming them into native KNIME nodes.

## Current Nodes

* **Auto-SARIMA Learner**: Automatically finds the optimal parameters for and trains a Seasonal AutoRegressive Integrated Moving Average (SARIMA) model on a given time series using simulated annealing optimization. Returns in-sample predictions, residuals, comprehensive model diagnostics, and optimization history. Features intelligent fallback strategies and adaptive diagnostic testing.

* **Auto-SARIMA Predictor**: Generates future forecasts using a pre-trained SARIMA model from the Auto-SARIMA Learner node.

* **Correspondence Analysis**: Performs Correspondence Analysis (CA) or Multiple Correspondence Analysis (MCA) on categorical data to explore associations between variables. Automatically selects the appropriate method based on input dimensionality.

* **Factor Analyzer**: Trains a dimensionality-reduction model using either Principal Component Analysis (PCA) or Exploratory Factor Analysis (EFA). Supports optional component/factor rotation for interpretability (varimax, promax, quartimax) and incremental PCA for large datasets.

* **Factor Scorer**: Applies a trained PCA or EFA model to new datasets, projecting data onto the learned components or factors. Maintains full mathematical consistency with all model variants including rotated solutions and incremental PCA.

## Package Organization

* **`icons/`**: Directory containing visual assets and icon images for the extension nodes, including specialized icons for time series analysis, correspondence analysis, and factor analysis components.
* **`config.yml`**: Configuration file specifying the path to the extension source code directory. This file works in conjunction with `knime.yml` to define the extension structure and dependencies.
* **`knime.yml`**: YAML configuration file containing extension metadata, including extension identification, module paths, and KNIME integration specifications.
* **`pixi.toml` & `pixi.lock`**: Pixi package manager configuration files defining Python dependencies, environment setup, and reproducible package management for the extension.
* **`src/social_science_ext.py`**: Main extension module that registers and initializes all nodes within the Social Science Extension for KNIME Analytics Platform.
* **`src/nodes/`**: Core implementation directory containing individual node modules:
  - `arima_learner.py`: SARIMA/SARIMAX time series model fitting node with simulated annealing optimization and comprehensive diagnostics
  - `arima_predictor.py`: Time series forecasting predictor node for applying trained ARIMA models
  - `correspondence_analysis.py`: Correspondence Analysis (CA) and Multiple Correspondence Analysis (MCA) implementation for categorical data analysis
  - `factor_analyzer.py`: Factor analysis learner with multiple rotation methods using factor_analyzer library
  - `factor_scorer.py`: Factor scoring node for applying trained models to new data with proper mathematical consistency
* **`src/util/`**: Utility module directory containing helper functions and shared code components:
  - `utils.py`: Collection of utility functions providing common data processing and validation methods

This extension provides a comprehensive suite of statistical analysis tools specifically designed for social science research within the KNIME Analytics Platform, supporting both time series analysis and multivariate statistical methods for categorical and continuous data.
