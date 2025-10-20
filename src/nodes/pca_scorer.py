import logging
import knime.extension as knext
import social_science_ext

LOGGER = logging.getLogger(__name__)

@knext.node(
    name="Factors Scorer",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="correspondence.png",
    category=social_science_ext.main_category,
    id="pca_scorer",
)
@knext.input_table(
    name="Input Data",
    description="Table containing the data to be transformed using the trained PCA model.",
)
@knext.input_binary(
    name="Model",
    description="Pickled PCA model object from the PCA Analysis node.",
    id="pca_analysis.model",
)
@knext.output_table(
    name="PCA Scores",
    description="Transformed data (principal component scores) for the input data.",
)
class PCAPredictorNode:
    """
    A KNIME predictor node that applies a trained PCA model to new data and outputs the principal component scores.

    **Model Overview:**
    This node takes a pickled PCA model (produced by the PCA Analysis node) and applies it to new input data to compute principal component scores. It supports models with or without rotation (Varimax, Promax, Quartimax, or none), and automatically standardizes input data if the model was trained with standardization.

    - **Input Data:** Must contain all numeric columns used during PCA training.
    - **Model Input:** Pickled object containing the trained PCA estimator, rotation matrix, scaling parameters, and feature column names.

    **Scoring Process:**
    1. Checks that all required feature columns are present in the input data.
    2. Drops rows with missing values in any feature column.
    3. Standardizes features using the mean and scale from the training data (if applicable).
    4. Applies the trained PCA transformation to obtain unrotated principal component scores.
    5. Applies the saved rotation matrix (if any) to the whitened scores:
        - **No Rotation:** Outputs unrotated, whitened scores.
        - **Varimax/Quartimax:** Applies orthogonal rotation.
        - **Promax:** Applies oblique rotation (scores may be correlated).
    6. Outputs the requested number of principal component scores as new columns.

    **Output Table:**
    - All original columns from the input data.
    - One column per requested principal component (e.g., PC1, PC2, ...).

    **Computational Details:**
    - Ensures the number of output components does not exceed the number trained in the model.
    - Uses the same scaling and rotation as the training node for consistency.
    - Handles both orthogonal and oblique rotations.

    **References:**
    - Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
    - Abdi, H., & Williams, L. J. (2010). Principal component analysis. *Wiley Interdisciplinary Reviews: Computational Statistics*, 2(4), 433–459.
    - Kaiser, H. F. (1958). The varimax criterion for analytic rotation in factor analysis. *Psychometrika*, 23(3), 187–200.
    - Harman, H. H. (1976). *Modern Factor Analysis* (3rd ed.). University of Chicago Press.
    """

    n_components = knext.IntParameter(
        label="Number of Components to output",
        description="Specify how many principal components to compute.",
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
        Defines the output schema by appending principal component columns to the input schema.
        """
        input_column_names = [col.name for col in input_schema]
        input_column_types = [col.ktype for col in input_schema]

        pc_column_names = [f"PC{i + 1}" for i in range(self.n_components)]
        pc_column_types = [knext.double()] * self.n_components

        all_column_names = input_column_names + pc_column_names
        all_column_types = input_column_types + pc_column_types

        return (knext.Schema(all_column_types, all_column_names),)

    def execute(self, exec_context, input_table, model_binary):
        """
        Applies the trained PCA model to the input data and outputs principal component scores.

        Parameters
        ----------
        exec_context : knext.ExecutionContext
            The KNIME execution context.
        input_table : knext.Table
            The input data table.
        model_binary : bytes
            The pickled PCA model object from the PCA Analysis node.

        Returns
        -------
        knext.Table
            The input data with appended principal component score columns.
        """
        # Import heavy dependencies only when needed
        import pickle
        import pandas as pd
        import numpy as np
        
        # Load the trained model and parameters
        model_data = pickle.loads(model_binary)
        scaler_mean = model_data["scaler_mean"]
        scaler_scale = model_data["scaler_scale"]
        features_cols = model_data["features_cols"]
        rotation_matrix = model_data.get("rotation_matrix")
        pca = model_data["pca"]

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
        trained_n_components = pca.n_components_
        if trained_n_components < max_dims:
            raise ValueError(
                f"Requested {max_dims} components, but the PCA model was trained with only {trained_n_components} components."
            )

        # Standardize using training scaler parameters if available
        if scaler_mean is not None and scaler_scale is not None:
            x_scaled = (X.values - scaler_mean) / scaler_scale
        else:
            x_scaled = X.values

        # Compute unrotated PCA scores
        scores_unrot = pca.transform(x_scaled)  # shape: (n_samples, n_fitted_components)
        scores_unrot = scores_unrot[:, :max_dims]

        # Eigenvalues of unrotated components (variances of scores_unrot)
        eigvals = np.array(pca.explained_variance_[:max_dims], dtype=float)
        eigvals[eigvals <= 0] = np.finfo(float).eps  # Defensive guard

        # Whiten: make unrotated scores unit-variance and uncorrelated
        s_whitened = scores_unrot / np.sqrt(eigvals)

        # Apply rotation if present
        if rotation_matrix is None:
            # NO_ROTATION: already uncorrelated with unit variance
            scores = s_whitened
        else:
            # Apply rotation matrix (works for both orthogonal and oblique rotations)
            rotation_matrix = rotation_matrix[:s_whitened.shape[1], :max_dims]
            scores = s_whitened @ rotation_matrix

        # Build output table
        score_columns = [f"PC{i + 1}" for i in range(max_dims)]
        scores_df = pd.DataFrame(scores, columns=score_columns)
        final_df = pd.concat([result_df, scores_df], axis=1)
        scores_df = scores_df.reset_index(drop=True)

        return knext.Table.from_pandas(final_df)
