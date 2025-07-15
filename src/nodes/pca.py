import logging
import knime.extension as knext
import pandas as pd
import numpy as np
from util import utils as kutil
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

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
    name="Principal Component Analysis",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/models/pca.png",
    category=kutil.category_multivariate_analysis,
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
        loadings = pca.components_.T.copy()
        phi = loadings[:, :max_dims]

        # Apply rotation if selected
        rotation_method = self.rotation_settings.rotation_method
        print(f"Applying rotation method: {rotation_method}")
        if rotation_method == "NO_ROTATION":
            rotated_loadings = phi
            rotation_matrix = np.eye(phi.shape[1])
        # default if no rotation

        elif rotation_method == "VARIMAX":
            rotated_loadings, rotation_matrix = self._varimax(phi)

        elif rotation_method == "PROMAX":
            rotated_loadings, rotation_matrix = self._promax(phi)

        else:
            raise ValueError(f"Unknown rotation method selected: {rotation_method}")

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
    
    def __init__(self, normalize=True, power=4, max_iter=1000, tol=1e-6):
        self.normalize = normalize
        self.power = power
        self.max_iter = max_iter
        self.tol = tol

    def _varimax(self, loadings):
        """
        Perform varimax (orthogonal) rotation, with optional Kaiser normalization.

        Parameters
        ----------
        loadings : array-like
            The loading matrix.

        Returns
        -------
        loadings : :obj:`numpy.ndarray`, shape (``n_features``, ``n_factors``)
            The loadings matrix.
        rotation_mtx : :obj:`numpy.ndarray`, shape (``n_factors``, ``n_factors``)
            The rotation matrix.
        """
        X = loadings.copy()
        n_rows, n_cols = X.shape
        if n_cols < 2:
            return X

        # normalize the loadings matrix
        # using sqrt of the sum of squares (Kaiser)
        if self.normalize:
            normalized_mtx = np.apply_along_axis(
                lambda x: np.sqrt(np.sum(x**2)), 1, X.copy()
            )
            X = (X.T / normalized_mtx).T

        # initialize the rotation matrix
        # to N x N identity matrix
        rotation_mtx = np.eye(n_cols)

        d = 0
        for _ in range(self.max_iter):
            old_d = d

            # take inner product of loading matrix
            # and rotation matrix
            basis = np.dot(X, rotation_mtx)

            # transform data for singular value decomposition using updated formula :
            # B <- t(x) %*% (z^3 - z %*% diag(drop(rep(1, p) %*% z^2))/p)
            diagonal = np.diag(np.squeeze(np.repeat(1, n_rows).dot(basis**2)))
            transformed = X.T.dot(basis**3 - basis.dot(diagonal) / n_rows)

            # perform SVD on
            # the transformed matrix
            U, S, V = np.linalg.svd(transformed)

            # take inner product of U and V, and sum of S
            rotation_mtx = np.dot(U, V)
            d = np.sum(S)

            # check convergence
            if d < old_d * (1 + self.tol):
                break

        # take inner product of loading matrix
        # and rotation matrix
        X = np.dot(X, rotation_mtx)
        
        # de-normalize the data
        if self.normalize:
            X = X.T * normalized_mtx
        else:
            X = X.T
        # convert loadings matrix to data frame
        loadings = X.T.copy()
        return loadings, rotation_mtx
    
    def _promax(self, phi, power=None):
        import numpy as np
        from sklearn.linear_model import LinearRegression

        if power is None:
            power = self.power

        X = phi.copy()
        n_rows, n_cols = X.shape
        if n_cols < 2:
            return X

        # Apply Kaiser normalization if enabled
        if self.normalize:
            row_norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / row_norms

        # First get orthogonal varimax rotation
        orthogonal = self._varimax(X)

        # Create target matrix
        target = np.sign(orthogonal) * (np.abs(orthogonal) ** power)

        # Linear regression without intercept
        model = LinearRegression(fit_intercept=False)
        model.fit(orthogonal, target)
        coef = model.coef_.T

        # Apply transformation
        oblique = np.dot(orthogonal, coef)

        # De-normalize if Kaiser was applied
        if self.normalize:
            oblique = oblique * row_norms

        return oblique

