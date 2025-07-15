import logging
import knime.extension as knext
import pandas as pd
import numpy as np
from util import utils as kutil
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle

LOGGER = logging.getLogger(__name__)

@knext.node(
    name="PCA Scorer",
    node_type=knext.NodeType.PREDICTOR,
    icon_path="icons/models/pca.png",
    category=kutil.category_multivariate_analysis,
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
        # Take all columns exactly as they are
        input_column_names = [col.name for col in input_schema]
        input_column_types = [col.ktype for col in input_schema]

        # Define new PC score columns
        pc_column_names = [f"PC{i + 1}" for i in range(self.n_components)]
        pc_column_types = [knext.double()] * self.n_components
        print("column_types:", pc_column_types)

        # Combine for output schema
        all_column_names = input_column_names + pc_column_names
        all_column_types = input_column_types + pc_column_types

        return (knext.Schema(all_column_types, all_column_names),)

    def execute(self, exec_context, input_table, model_binary):

        # 1️ Load the trained model data, table and parameters
        model_data = pickle.loads(model_binary)
        loadings = model_data["loadings"]
        scaler_mean = model_data["scaler_mean"]
        scaler_scale = model_data["scaler_scale"]
        features_cols = model_data["features_cols"]
        rotation_matrix = model_data.get("rotation_matrix")
        unrotated_components = model_data.get("unrotated_components")
        max_dims = self.n_components

        df = input_table.to_pandas()

        # 2️ Check all feature columns exist in input
        missing_columns = [col for col in features_cols if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Input table is missing required columns: {missing_columns}")

        # 2️ Build mask for complete feature rows
        features_complete_mask = df[features_cols].notna().all(axis=1)

        # 3 Filter feature matrix and all columns consistently
        X = df.loc[features_complete_mask, features_cols].reset_index(drop=True)
        result_df = df.loc[features_complete_mask].reset_index(drop=True).copy()

        # 4 Check if trained model has enough components
        trained_n_components = loadings.shape[1]
        if trained_n_components < max_dims:
            raise ValueError(f"Requested {max_dims} components, but the PCA model was trained with only {trained_n_components} components.")

        # 5 Standardize using training scaler parameters if available
        if scaler_mean is not None and scaler_scale is not None:
            x_scaled = (X.values - scaler_mean) / scaler_scale
        else:
            x_scaled = X.values

        # 6 Cut off loadings to match requested dimensions
        if loadings.shape[1] < max_dims:
            raise ValueError(f"Requested {max_dims} components, but the PCA model has only {loadings.shape[1]} components.")
        
        loadings = loadings[:, :max_dims]
        scores_full = np.dot(x_scaled, loadings)

        # Ensure scores are limited to requested dimensions
        scores = scores_full[:, :max_dims]

        # 8 Build output table
        score_columns = [f"PC{i + 1}" for i in range(max_dims)]
        scores_df = pd.DataFrame(scores, columns=score_columns)
        final_df = pd.concat([result_df, scores_df], axis=1)
        print("SCORES SHAPE:", scores_df.shape)
        print("X SHAPE:", X.shape)
        scores_df = scores_df.reset_index(drop=True)

        return knext.Table.from_pandas(final_df)
