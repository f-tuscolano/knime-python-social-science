import logging
import knime.extension as knext
from util import utils as kutil
import social_science_ext

LOGGER = logging.getLogger(__name__)

@knext.parameter_group(label="Rotation Settings",)
class RotationSettings:
    """
    These parameters control the rotation method applied to the PCA results.
    """
    class RotationMethods(knext.EnumParameterOptions):
        NO_ROTATION = (
            "None",
            "Do not apply any rotation; use the raw PCA loadings."
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
        description="Choose a rotation method to apply to the PCA results.",
        default_value=RotationMethods.NO_ROTATION.name,
        enum=RotationMethods,
    )

class PcaMethod(knext.EnumParameterOptions):
        STANDARD = ("Standard PCA", "Standard Principal Component Analysis.")
        INCREMENTAL = ("Incremental PCA", "Incremental Principal Component Analysis (IncrementalPCA) for large datasets.")

@knext.node(
    name="Factor Analyzer",
    node_type=knext.NodeType.LEARNER,
    icon_path="correspondence.png",
    category=social_science_ext.main_category,
    id="pca_analysis",
)
@knext.input_table(
    name="Input Data",
    description="Table containing one or more numeric columns for PCA.",
)
@knext.output_table(
    name="Explained Variance",
    description="Eigenvalues and variance ratios for each principal component.",
)
@knext.output_table(
    name="Component Loadings",
    description="Loadings of each input variable on the principal components.",
)
@knext.output_binary(
    name="Model",
    description="Pickled PCA model object that can be used by the PCA Predictor node to transform new data.",
    id="pca_analysis.model",
)

class PCAAnalysisNode:
    """
    A KNIME learner node that performs Principal Component Analysis (PCA) on numeric input data.
    This node is designed to extract latent structure from multivariate numeric datasets by projecting them into a lower-dimensional space. It supports several rotation methods to enhance interpretability of the principal components.

    **Model Overview:**
    This node computes principal components that capture the maximum variance in the data, optionally standardizing features before analysis. It supports both standard and incremental PCA algorithms for scalability.

    - **PCA**: Reduces dimensionality by finding orthogonal axes (principal components) that explain the most variance.
    - **Rotation**: Improves interpretability of the component loadings by applying orthogonal or oblique rotations (Varimax, Promax, Quartimax).

    The node outputs:
    - Eigenvalues and explained variance ratios for each principal component.
    - Rotated component loadings for each input variable.
    - A pickled model object for downstream transformation of new data.

    **Component Loadings Table:**
    - **Variable**: Name of the input feature.
    - **Loading (PC#)**: The loading of the variable on each principal component after rotation.

    **Rotation Methods:**
    - **None**: No rotation; raw PCA loadings are used.
    - **Varimax**: Orthogonal rotation maximizing variance of squared loadings (simplifies columns).
    - **Promax**: Oblique rotation allowing correlated components (applies Varimax, then raises loadings to a power).
    - **Quartimax**: Orthogonal rotation maximizing variance of squared loadings across variables (simplifies rows).

    **Computational Details:**
    - Handles missing values by dropping rows with NaNs in selected columns.
    - Automatically limits the number of components to the number of selected features.
    - Flips the sign of loadings for each component if the sum is less than 1, for consistency.
    - Stores the fitted PCA model, rotation matrix, and scaling parameters for later use.

    **References:**
    - Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
    - Abdi, H., & Williams, L. J. (2010). Principal component analysis. *Wiley Interdisciplinary Reviews: Computational Statistics*, 2(4), 433–459.
    """
    pca_method = knext.EnumParameter(
        label="PCA Method",
        description="Select the PCA algorithm to use.",
        default_value=PcaMethod.STANDARD.name,
        enum=PcaMethod,
    )
    n_components = knext.IntParameter(
        label="Number of Components",
        description="Specify how many principal components to compute.",
        default_value=2,
        min_value=1,
        max_value=1000,
    )
    
    rotation_settings = RotationSettings()
    
    features_cols = knext.MultiColumnParameter(
        label="Numeric Input Columns",
        description="Select two or more numeric columns to include in the PCA.",
        column_filter=kutil.is_numeric,
    )
    standardize_column = knext.BoolParameter(
        label="Standardize input data",
        description="Optionally standardize the input data before applying PCA.",
        default_value=True,
    )

    def configure(self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema):
        num_cols = len(self.features_cols)
        max_dims = min(self.n_components, num_cols)

        variance_schema = knext.Schema(
            [knext.double(), knext.double(), knext.double()],
            ["Eigenvalue", "Explained Variance Ratio", "Cumulative Explained Variance"]
        )

        loadings_schema = knext.Schema(
            [knext.string()] + [knext.double()] * max_dims,
            ["Variable"] + [f"Loading (PC{i+1})" for i in range(max_dims)]
        )
        # Define the binary model output port schema
        binary_model_schema = knext.BinaryPortObjectSpec("pca_analysis.model")

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
            raise ValueError("PCA requires at least two rows and two numeric columns.")

        max_dims = min(self.n_components, X.shape[1])
        if max_dims < 2:
            raise knext.InvalidParametersError("Please select at least two numeric columns for PCA.")

        # Select PCA method
        method = self.pca_method
        if method == PcaMethod.STANDARD.name:
            pca = PCA(n_components=max_dims)
            if self.standardize_column:
                scaler = StandardScaler()
                X_pca = scaler.fit_transform(X)
            else:
                scaler = None
                X_pca = X.values
        elif method == PcaMethod.INCREMENTAL.name:
            from sklearn.decomposition import IncrementalPCA
            pca = IncrementalPCA(n_components=max_dims)
            if self.standardize_column:
                scaler = StandardScaler()
                X_pca = scaler.fit_transform(X)
            else:
                scaler = None
                X_pca = X.values
        else:
            raise ValueError("Unknown PCA method selected.")

        # Fit PCA
        pca.fit(X_pca)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
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
        if hasattr(pca, 'explained_variance_'):
            eigenvalues = pca.explained_variance_[:max_dims]
        else:
            eigenvalues = np.var(pca.transform(X_pca), axis=0, ddof=1)[:max_dims]
        if hasattr(pca, 'explained_variance_ratio_'):
            var_ratio = pca.explained_variance_ratio_[:max_dims]
            cum_var = np.cumsum(var_ratio)
        else:
            var_ratio = eigenvalues / np.sum(eigenvalues)
            cum_var = np.cumsum(var_ratio)

        variance_df = pd.DataFrame({
            "Eigenvalue": eigenvalues,
            "Explained Variance Ratio": var_ratio,
            "Cumulative Explained Variance": cum_var,
        })

        # Prepare loadings table
        loadings_df = pd.DataFrame(
            rotated_loadings[:, :max_dims],
            index=self.features_cols,
            columns=[f"Loading (PC{i+1})" for i in range(max_dims)]
        ).reset_index().rename(columns={"index": "Variable"})

        # Save the trained PCA model to the binary output port
        if self.standardize_column:
            scaler_mean = scaler.mean_
            scaler_scale = scaler.scale_
        else:
            scaler_mean = None
            scaler_scale = None

        model_binary = pickle.dumps({
            "pca": pca,
            "loadings": rotated_loadings,
            "rotation_matrix": rotation_matrix,
            "unrotated_components": phi,
            "scaler_mean": scaler_mean,
            "scaler_scale": scaler_scale,
            "features_cols": self.features_cols,
        })

        return (
            knext.Table.from_pandas(variance_df),
            knext.Table.from_pandas(loadings_df),
            model_binary,
        )