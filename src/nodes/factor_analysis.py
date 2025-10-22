import logging
import knime.extension as knext
from util import utils as kutil
import social_science_ext

LOGGER = logging.getLogger(__name__)

@knext.parameter_group(label="Rotation Settings",)
class RotationSettings:
    """
    These parameters control the rotation method applied to the analysis results.
    """
    class RotationMethods(knext.EnumParameterOptions):
        NO_ROTATION = (
            "None",
            "Do not apply any rotation; use the raw loadings."
        )
        VARIMAX = (
            "Varimax",
            "An orthogonal rotation that maximizes the variance of squared loadings, so that each component loads highly on a small number of variables."
        )
        PROMAX = (
            "Promax",
            "An oblique rotation that first performs Varimax, then raises loadings to a power to allow correlated components."
        )
        QUARTIMAX = (
            "Quartimax",
            "An orthogonal rotation that maximizes the variance of squared loadings across variables."
        )
    rotation_method = knext.EnumParameter(
        label="Select a rotation method",
        description="Choose a rotation method to apply to the analysis results.",
        default_value=RotationMethods.NO_ROTATION.name,
        enum=RotationMethods,
    )

class AnalysisMethod(knext.EnumParameterOptions):
        STANDARD = ("Standard PCA", "Standard Principal Component Analysis.")
        INCREMENTAL = ("Incremental PCA", "Incremental Principal Component Analysis (IncrementalPCA) for large datasets.")
        FACTOR_ANALYSIS = ("Exploratory Factor Analysis", "Exploratory Factor Analysis using maximum likelihood estimation to find latent factors.")

@knext.node(
    name="Factor Analyzer",
    node_type=knext.NodeType.LEARNER,
    icon_path="correspondence.png",
    category=social_science_ext.main_category,
    id="factor_analysis",
)
@knext.input_table(
    name="Input Data",
    description="Table containing one or more numeric columns for factor analysis.",
)
@knext.output_table(
    name="Model fit",
    description="Model fit statistics: eigenvalues and variance ratios for PCA methods, log-likelihood for Factor Analysis.",
)
@knext.output_table(
    name="Component Loadings",
    description="Loadings of each input variable on the principal components, including communalities (explained variance) and noise variance (unique variance).",
)
@knext.output_binary(
    name="Model",
    description="Pickled factor analysis model object that can be used by predictor nodes to transform new data.",
    id="factor_analysis.model",
)

class FactorAnalysisNode:
    """
    A KNIME learner node that performs Principal Component Analysis (PCA) or Exploratory Factor Analysis (EFA) on numeric input data.
    This node is designed to extract latent structure from multivariate numeric datasets by projecting them into a lower-dimensional space. It supports several rotation methods to enhance interpretability of the components or factors.

    **Model Overview:**
    This node supports three analysis methods:
    
    - **Standard PCA**: Reduces dimensionality by finding orthogonal axes (principal components) that explain the most variance.
    - **Incremental PCA**: Memory-efficient PCA for large datasets that cannot fit in memory.
    - **Exploratory Factor Analysis**: Uses maximum likelihood estimation to identify latent factors that explain correlations among observed variables.

    **Rotation**: Improves interpretability of the component/factor loadings by applying orthogonal or oblique rotations (Varimax, Promax, Quartimax).

    The node outputs:
    - Model fit statistics including eigenvalues for PCA methods and log-likelihood for Factor Analysis.
    - Rotated component/factor loadings with communalities and noise variance for each input variable.
    - A pickled model object for downstream transformation of new data.

    **Model Fit Table:**
    - **Dimension**: Component/factor number (1, 2, 3... for PCA; selected number of factors for Factor Analysis).
    - **Eigenvalue**: Variance explained by each component (PCA methods only; 0 for Factor Analysis).
    - **Explained Variance Ratio**: Proportion of total variance explained by each component (PCA methods only; 0 for Factor Analysis).
    - **Cumulative Explained Variance**: Cumulative proportion of variance explained (PCA methods only; 0 for Factor Analysis).
    - **Log-Likelihood**: Maximum likelihood value for model goodness-of-fit (Factor Analysis only; 0 for PCA methods).

    **Component Loadings Table:**
    - **Variable**: Name of the input feature.
    - **Communalities**: Sum of squared loadings across all factors; indicates how much variance in each variable is explained by the factor solution (range 0-1).
    - **Noise Variance**: Complement of communalities (1 - communalities); represents the unique variance in each variable not explained by the common factors.
    - **Loading (PC#)**: The loading of the variable on each principal component/factor after rotation.

    **Rotation Methods:**
    - **None**: No rotation; raw loadings are used.
    - **Varimax**: Orthogonal rotation maximizing variance of squared loadings (simplifies columns).
    - **Promax**: Oblique rotation allowing correlated components (applies Varimax, then raises loadings to a power).
    - **Quartimax**: Orthogonal rotation maximizing variance of squared loadings across variables (simplifies rows).

    **Variance Components Interpretation:**
    - **Communalities**: High values (>0.7) indicate variables well-represented by the factor solution; low values (<0.3) suggest need for additional factors.
    - **Noise Variance**: High values (>0.7) indicate substantial unique variance; low values (<0.3) suggest the variable is well-explained by common factors.
    - **For PCA**: Communalities often close to 1.0, noise variance close to 0.0 since PCA explains maximum variance.
    - **For Factor Analysis**: More moderate values expected, representing the distinction between common and unique variance.

    **Computational Details:**
    - Handles missing values by dropping rows with NaNs in selected columns.
    - Automatically limits the number of components to the number of selected features.
    - Flips the sign of loadings for each component if the sum is less than 1, for consistency.
    - Stores the fitted model, rotation matrix, and scaling parameters for later use.

    **Eigenvalue Computation:**
    - **PCA Methods**: Uses true eigenvalues from covariance matrix decomposition.
    - **Factor Analysis**: Computes pseudo-eigenvalues as sum of squared loadings per factor (Σ(loading_ik²)).
      These represent the variance captured by each factor and are analogous to eigenvalues in interpretation.
      This approach follows standard factor analysis practice for assessing factor importance.

    **Model Output for Scoring:**
    The binary model output contains comprehensive information for downstream prediction:
    - Fitted model object (sklearn PCA/IncrementalPCA/FactorAnalysis)
    - Rotated and unrotated loadings matrices
    - Rotation transformation matrix and method
    - Preprocessing parameters (standardization means/scales)
    - Variance statistics (eigenvalues, explained variance ratios)
    - Feature column names and analysis method identifier

    **References:**
    - Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
    - Abdi, H., & Williams, L. J. (2010). Principal component analysis. *Wiley Interdisciplinary Reviews: Computational Statistics*, 2(4), 433–459.
    - Bartholomew, D. J., et al. (2011). *Analysis of Multivariate Social Science Data* (2nd ed.). Chapman and Hall/CRC.
    - Fabrigar, L. R., & Wegener, D. T. (2012). *Exploratory Factor Analysis*. Oxford University Press.
    """
    analysis_method = knext.EnumParameter(
        label="Analysis Method",
        description="Select the dimensionality reduction or factor analysis algorithm to use.",
        default_value=AnalysisMethod.STANDARD.name,
        enum=AnalysisMethod,
    )
    n_components = knext.IntParameter(
        label="Number of Components",
        description="Specify how many principal components to compute.",
        default_value=2,
        min_value=1,
        max_value=1000,
    )
    
    features_cols = knext.MultiColumnParameter(
        label="Numeric Input Columns",
        description="Select two or more numeric columns to include in the factor analysis.",
        column_filter=kutil.is_numeric,
    )
    
    standardize_column = knext.BoolParameter(
        label="Standardize input data",
        description="Optionally standardize the input data before applying factor analysis.",
        default_value=True,
    )
    
    rotation_settings = RotationSettings()

    def configure(self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema):
        num_cols = len(self.features_cols)
        max_dims = min(self.n_components, num_cols)

        variance_schema = knext.Schema(
            [knext.double(), knext.double(), knext.double(), knext.double(), knext.double()],
            ["Dimension", "Eigenvalue", "Explained Variance Ratio", "Cumulative Explained Variance", "Log-Likelihood"]
        )

        loadings_schema = knext.Schema(
            [knext.string(), knext.double(), knext.double()] + [knext.double()] * max_dims,
            ["Variable", "Communalities", "Noise Variance"] + [f"Loading (PC{i+1})" for i in range(max_dims)]
        )
        # Define the binary model output port schema
        binary_model_schema = knext.BinaryPortObjectSpec("factor_analysis.model")

        return (
            variance_schema,
            loadings_schema,
            binary_model_schema,
        )

    def execute(self, exec_context: knext.ExecutionContext, input_table: knext.Table):
        # Import heavy dependencies only when needed
        import pickle
        import pandas as pd
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from factor_analyzer.rotator import Rotator
        
        df = input_table.to_pandas()
        X = df[self.features_cols].dropna()

        if X.shape[0] < 2 or X.shape[1] < 2:
            raise ValueError("Factor analysis requires at least two rows and two numeric columns.")

        max_dims = min(self.n_components, X.shape[1])

        # Select analysis method
        method = self.analysis_method
        if method == AnalysisMethod.STANDARD.name:
            factor_model = PCA(n_components=max_dims)
            if self.standardize_column:
                scaler = StandardScaler()
                x_transformed = scaler.fit_transform(X)
            else:
                scaler = None
                x_transformed = X.values
        elif method == AnalysisMethod.INCREMENTAL.name:
            from sklearn.decomposition import IncrementalPCA
            factor_model = IncrementalPCA(n_components=max_dims)
            if self.standardize_column:
                scaler = StandardScaler()
                x_transformed = scaler.fit_transform(X)
            else:
                scaler = None
                x_transformed = X.values
        elif method == AnalysisMethod.FACTOR_ANALYSIS.name:
            from sklearn.decomposition import FactorAnalysis
            factor_model = FactorAnalysis(n_components=max_dims, svd_method='lapack')
            if self.standardize_column:
                scaler = StandardScaler()
                x_transformed = scaler.fit_transform(X)
            else:
                scaler = None
                x_transformed = X.values
        else:
            raise ValueError("Unknown analysis method selected.")

        # Fit the analysis model
        factor_model.fit(x_transformed)
        
        # Calculate loadings based on method
        if method == AnalysisMethod.FACTOR_ANALYSIS.name:
            # For Factor Analysis, use the components directly as loadings
            loadings = factor_model.components_.T
        else:
            # For PCA methods, scale loadings by sqrt of explained variance
            loadings = factor_model.components_.T * np.sqrt(factor_model.explained_variance_)
        
        phi = loadings[:, :max_dims]

        # Apply rotation if selected
        rotation_method = self.rotation_settings.rotation_method
        print(f"Applying rotation method: {rotation_method}")
        
        # Get the actual enum value (display name)
        rotation_value = RotationSettings.RotationMethods[rotation_method].value[0]
        print(f"Rotation value: {rotation_value}")
        
        if rotation_value == "None":
            rotated_loadings = phi
            rotation_matrix = np.eye(phi.shape[1])
        else:
            # Use factor_analyzer's Rotator - convert to lowercase for the library
            rotation_method_lower = rotation_value.lower()
            rotator = Rotator(method=rotation_method_lower)
            rotated_loadings = rotator.fit_transform(phi)
            rotation_matrix = rotator.rotation_

        # Flip sign of loadings if the sum over the same dimension is < 1
        # (for each component/dimension)
        for i in range(rotated_loadings.shape[1]):
            col_sum = np.sum(rotated_loadings[:, i])
            if col_sum < 1:
                rotated_loadings[:, i] *= -1
                # If rotation_matrix exists and is square, flip its sign for the same component
                if rotation_matrix.shape[0] == rotation_matrix.shape[1]:
                    rotation_matrix[:, i] *= -1

        # Define eigenvalues, explained variance ratio, and cumulative explained variance
        if method == AnalysisMethod.FACTOR_ANALYSIS.name:
            # For Exploratory Factor Analysis (EFA), leave eigenvalue table empty
            # Factor Analysis uses maximum likelihood estimation and eigenvalues are not directly meaningful
            eigenvalues = np.array([])
            var_ratio = np.array([])
            cum_var = np.array([])
        elif hasattr(factor_model, 'explained_variance_'):
            # For PCA methods: use true eigenvalues from the covariance matrix decomposition
            eigenvalues = factor_model.explained_variance_[:max_dims]
            var_ratio = factor_model.explained_variance_ratio_[:max_dims]
            cum_var = np.cumsum(var_ratio)
        else:
            # Fallback: compute eigenvalues from transformed data variance
            eigenvalues = np.var(factor_model.transform(x_transformed), axis=0, ddof=1)[:max_dims]
            var_ratio = eigenvalues / np.sum(eigenvalues)
            cum_var = np.cumsum(var_ratio)
        
        # Create the variance DataFrame 
        if len(eigenvalues) > 0:
            # For PCA methods: create dimension counter from 1 to max_dims
            dimension_col = np.arange(1, len(eigenvalues) + 1, dtype=np.float64)
            log_likelihood_col = np.zeros(len(eigenvalues), dtype=np.float64)  # Set to 0 for PCA
            variance_df = pd.DataFrame({
                "Dimension": dimension_col,
                "Eigenvalue": eigenvalues.astype(np.float64),
                "Explained Variance Ratio": var_ratio.astype(np.float64),
                "Cumulative Explained Variance": cum_var.astype(np.float64),
                "Log-Likelihood": log_likelihood_col,
            })
        else:
            # For Factor Analysis: single row with selected dimensions and log-likelihood
            # Get log-likelihood if available, otherwise use 0
            if hasattr(factor_model, 'loglike_'):
                # Handle both scalar and array cases
                loglike_raw = factor_model.loglike_
                if hasattr(loglike_raw, '__len__'):  # It's an array/list
                    log_likelihood_value = float(np.max(loglike_raw))
                else:  # It's a scalar
                    log_likelihood_value = float(loglike_raw)
            else:
                log_likelihood_value = 0.0
            
            variance_df = pd.DataFrame({
                "Dimension": np.array([max_dims], dtype=np.float64),           # Single value: selected number of components
                "Eigenvalue": np.array([0.0], dtype=np.float64),            # Set to 0 for Factor Analysis
                "Explained Variance Ratio": np.array([0.0], dtype=np.float64), # Set to 0 for Factor Analysis
                "Cumulative Explained Variance": np.array([0.0], dtype=np.float64), # Set to 0 for Factor Analysis
                "Log-Likelihood": np.array([log_likelihood_value], dtype=np.float64), # Single scalar log-likelihood
            })

        # Calculate communalities (sum of squared loadings for each variable)
        communalities = np.sum(rotated_loadings[:, :max_dims] ** 2, axis=1)
        
        # Calculate noise variance (unique variance not explained by factors)
        noise_variance = 1.0 - communalities
        
        # Prepare loadings table with communalities and noise variance
        loadings_data = pd.DataFrame(
            rotated_loadings[:, :max_dims],
            index=self.features_cols,
            columns=[f"Loading (PC{i+1})" for i in range(max_dims)]
        )
        
        # Add communalities and noise variance columns
        loadings_data.insert(0, "Communalities", communalities)
        loadings_data.insert(1, "Noise Variance", noise_variance)
        
        # Reset index to make Variable a column
        loadings_df = loadings_data.reset_index().rename(columns={"index": "Variable"})

        # Save the trained factor analysis model to the binary output port for scoring/prediction
        # This model object contains all necessary information for transforming new data
        if self.standardize_column:
            scaler_mean = scaler.mean_
            scaler_scale = scaler.scale_
        else:
            scaler_mean = None
            scaler_scale = None

        # Create comprehensive model dictionary for scoring node
        model_dict = {
            # Core model components
            "model": factor_model,                    # The fitted sklearn model (PCA/IncrementalPCA/FactorAnalysis)
            "analysis_method": method,                # Method identifier for proper reconstruction
            "n_components": max_dims,                 # Number of components/factors
            
            # Loadings and rotation information
            "loadings": rotated_loadings,             # Final rotated loadings matrix
            "rotation_matrix": rotation_matrix,       # Rotation transformation matrix
            "unrotated_components": phi,              # Original unrotated components
            "rotation_method": rotation_value,  # Rotation method used
            
            # Preprocessing information
            "scaler_mean": scaler_mean,               # Feature means (if standardized)
            "scaler_scale": scaler_scale,             # Feature scales (if standardized)
            "standardize_column": self.standardize_column,  # Whether standardization was applied
            "features_cols": self.features_cols,      # Original feature column names
            
        }

        model_binary = pickle.dumps(model_dict)

        return (
            knext.Table.from_pandas(variance_df),
            knext.Table.from_pandas(loadings_df),
            model_binary,
        )