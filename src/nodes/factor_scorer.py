import logging
import knime.extension as knext
import social_science_ext

LOGGER = logging.getLogger(__name__)

@knext.node(
    name="Factors Scorer",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="correspondence.png",
    category=social_science_ext.main_category,
    id="factor_scorer",
)
@knext.input_table(
    name="Input Data",
    description="Table containing the data to be transformed using the trained factor analysis model.",
)
@knext.input_binary(
    name="Model",
    description="Pickled factor analysis model object from the Factor Analyzer node.",
    id="factor_analysis.model",
)
@knext.output_table(
    name="Factor Scores",
    description="Transformed data (factor/component scores) for the input data.",
)
class FactorScorerNode:
    """
    A KNIME predictor node that applies a trained factor analysis model to new data and outputs factor/component scores.

    **Model Overview:**
    This node takes a pickled factor analysis model (produced by the Factor Analyzer node) and applies it to new input data to compute factor/component scores. It supports multiple analysis methods (Standard PCA, Incremental PCA, Exploratory Factor Analysis) with or without rotation (None, Varimax, Promax, Quartimax, Equamax), and automatically standardizes input data if the model was trained with standardization.

    - **Input Data:** Must contain all numeric columns used during model training.
    - **Model Input:** Comprehensive pickled object containing the trained model, rotation matrix, scaling parameters, feature column names, and analysis method information.

    **Scoring Process:**
    1. Checks that all required feature columns are present in the input data.
    2. Drops rows with missing values in any feature column.
    3. Standardizes features using the mean and scale from the training data (if applicable).
    4. Applies the trained model transformation to obtain unrotated scores.
    5. Applies the saved rotation matrix (if any) to the transformed scores:
        - **No Rotation:** Outputs unrotated scores.
        - **Varimax/Quartimax/Equamax:** Applies orthogonal rotation.
        - **Promax:** Applies oblique rotation (scores may be correlated).
    6. Outputs the requested number of factor/component scores as new columns.

    **Output Table:**
    - All original columns from the input data.
    - One column per requested factor/component (e.g., PC1, PC2, ...).

    **Computational Details:**
    - Ensures the number of output components does not exceed the number trained in the model.
    - Uses the same scaling and rotation as the training node for consistency.
    - Handles both orthogonal and oblique rotations.
    - Supports all analysis methods: Standard PCA, Incremental PCA, and Exploratory Factor Analysis.
    - For Factor Analysis, applies the same transformation logic but interprets output as factor scores.

    **References:**
    - Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
    - Abdi, H., & Williams, L. J. (2010). Principal component analysis. *Wiley Interdisciplinary Reviews: Computational Statistics*, 2(4), 433–459.
    - Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis. *Psychometrika*, 23(3), 187–200.
    - Harman, H. H. (1976). *Modern Factor Analysis* (3rd ed.). University of Chicago Press.
    - Bartholomew, D. J., et al. (2011). *Analysis of Multivariate Social Science Data* (2nd ed.). Chapman and Hall/CRC.
    """

    n_components = knext.IntParameter(
        label="Number of Components to output",
        description="Specify how many factors/components to compute.",
        default_value=2,
        min_value=1,
        max_value=1000,
    )

    def configure(
        self,
        configure_context: knext.ConfigurationContext,
        input_schema: knext.Schema,
        input_model: bytes,
    ):
        """
        Defines the output schema by appending factor/component columns to the input schema.
        """
        input_column_names = [col.name for col in input_schema]
        input_column_types = [col.ktype for col in input_schema]

        fc_column_names = [f"FC{i + 1}" for i in range(self.n_components)]
        fc_column_types = [knext.double()] * self.n_components

        all_column_names = input_column_names + fc_column_names
        all_column_types = input_column_types + fc_column_types

        return (knext.Schema(all_column_types, all_column_names),)

    def execute(self, exec_context, input_table, model_binary):
        """
        Applies the trained factor analysis model to the input data and outputs factor/component scores.

        Parameters
        ----------
        exec_context : knext.ExecutionContext
            The KNIME execution context.
        input_table : knext.Table
            The input data table.
        model_binary : bytes
            The pickled factor analysis model object from the Factor Analyzer node.

        Returns
        -------
        knext.Table
            The input data with appended factor/component score columns.
        """
        # Import heavy dependencies only when needed
        import pickle
        import pandas as pd
        import numpy as np
        
        # Load the trained model and parameters from the comprehensive model dictionary
        model_data = pickle.loads(model_binary)
        
        # Extract model components
        factor_model = model_data["model"]                    # The fitted sklearn model
        model_n_components = model_data["n_components"]       # Number of components trained
        
        # Extract preprocessing information
        scaler_mean = model_data["scaler_mean"]               # Feature means (if standardized)
        scaler_scale = model_data["scaler_scale"]             # Feature scales (if standardized)
        features_cols = model_data["features_cols"]           # Original feature column names
        
        # Extract rotation information
        rotation_matrix = model_data.get("rotation_matrix")   # Rotation transformation matrix
        rotation_method = model_data.get("rotation_method", "None")   # Rotation method used

        max_dims = self.n_components

        df = input_table.to_pandas()

        # Ensure all required feature columns are present
        missing_columns = [col for col in features_cols if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Input table is missing required columns: {missing_columns}")

        # Mask for complete feature rows
        features_complete_mask = df[features_cols].notna().all(axis=1)

        # Filter feature matrix and all columns consistently
        X = df.loc[features_complete_mask, features_cols].reset_index(drop=True)
        result_df = df.loc[features_complete_mask].reset_index(drop=True).copy()

        # Check if trained model has enough components
        if model_n_components < max_dims:
            raise ValueError(
                f"Requested {max_dims} components, but the model was trained with only {model_n_components} components."
            )

        # Standardize using training scaler parameters if available
        if scaler_mean is not None and scaler_scale is not None:
            x_scaled = (X.values - scaler_mean) / scaler_scale
        else:
            x_scaled = X.values

        # Compute unrotated scores using the appropriate model
        scores_unrot = factor_model.transform(x_scaled)  # shape: (n_samples, n_fitted_components)
        scores_unrot = scores_unrot[:, :max_dims]

        # Handle variance standardization based on analysis method
        analysis_method = model_data["analysis_method"]
        
        if rotation_method and analysis_method in ["STANDARD", "INCREMENTAL"]:
            # For PCA methods: use true eigenvalues (UNCHANGED - working correctly)
            eigvals = np.array(factor_model.explained_variance_[:max_dims], dtype=float)
            eigvals[eigvals <= 0] = np.finfo(float).eps  # Defensive guard
            # Whiten: make unrotated scores unit-variance and uncorrelated
            s_whitened = scores_unrot / np.sqrt(eigvals)
        elif rotation_method and analysis_method == "FACTOR_ANALYSIS":
            # For Factor Analysis: standardize by factor variances to ensure orthogonality
            # Compute factor variances from unrotated scores
            factor_variances = np.var(scores_unrot, axis=0, ddof=1)
            factor_variances[factor_variances <= 0] = np.finfo(float).eps  # Defensive guard
            # Whiten: make unrotated scores unit-variance for proper rotation
            s_whitened = scores_unrot / np.sqrt(factor_variances)
        else:
            # For no rotation: use scores as-is
            s_whitened = scores_unrot

        # Apply rotation if present
        if rotation_matrix is None or rotation_method == "None":
            # NO_ROTATION: use whitened scores as-is
            scores = s_whitened
        else:
            # For all methods: Apply rotation matrix to whitened scores
            # This ensures consistent handling and proper orthogonality
            rotation_matrix = rotation_matrix[:s_whitened.shape[1], :max_dims]
            scores = s_whitened @ rotation_matrix

        # Build output table with appropriate column naming
        # Use FC (Factor Component) as a general term that works for both PCA and EFA
        score_columns = [f"FC{i + 1}" for i in range(max_dims)]
        scores_df = pd.DataFrame(scores, columns=score_columns)
        final_df = pd.concat([result_df, scores_df], axis=1)

        return knext.Table.from_pandas(final_df)
