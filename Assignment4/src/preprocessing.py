import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def parse_arff_nominal_values(attribute_type: str) -> list[str] | None:
    """
    Parse ARFF nominal values from a declaration like:
    {Private, Self-emp-not-inc, Federal-gov}

    Returns None for numeric attributes.
    """
    attribute_type = attribute_type.strip()

    if attribute_type.lower() == "numeric":
        return None

    match = re.match(r"^\{(.+)\}$", attribute_type)

    if match is None:
        return None

    values = [value.strip() for value in match.group(1).split(",")]
    return values


def build_declared_categories(
    metadata: dict,
    categorical_columns: list[str],
    target_column: str,
) -> tuple[dict[str, list[str]], list[str]]:
    """
    Build stable category lists from the ARFF header.

    This is important because different train/test splits may not contain rare
    categories, but we still want the same one-hot dimensions across seeds.
    """
    attribute_map = {
        name: attr_type
        for name, attr_type in metadata["attributes"]
    }

    categorical_categories = {}

    for column in categorical_columns:
        values = parse_arff_nominal_values(attribute_map[column])

        if values is None:
            raise ValueError(f"Column '{column}' does not have nominal ARFF values.")

        if "Missing" not in values:
            values = values + ["Missing"]

        categorical_categories[column] = values

    target_values = parse_arff_nominal_values(attribute_map[target_column])

    if target_values is None:
        raise ValueError(f"Target column '{target_column}' does not have nominal ARFF values.")

    # Use a stable and intuitive order for the Adult target.
    if set(target_values) == {"<=50K", ">50K"}:
        target_values = ["<=50K", ">50K"]

    return categorical_categories, target_values


def create_one_hot_encoder(categories: list[list[str]]) -> OneHotEncoder:
    """
    Create OneHotEncoder while supporting both old and new scikit-learn versions.
    """
    try:
        return OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=False,
        )
    except TypeError:
        return OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse=False,
        )


class AdultPreprocessor:
    """
    Preprocessor for the Adult dataset.

    It performs:
    1. Median imputation for numerical features.
    2. MinMax scaling of numerical features to [-1, 1].
    3. Missing-value replacement for categorical features.
    4. One-hot encoding for categorical features.
    5. One-hot encoding for the target label.

    The final GAN vector is:
        [processed_numeric_features,
         processed_categorical_features,
         one_hot_target]
    """

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: list[str],
        target_column: str,
        categorical_categories: dict[str, list[str]],
        target_categories: list[str],
    ) -> None:
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.target_column = target_column
        self.categorical_categories = categorical_categories
        self.target_categories = target_categories

        self.numeric_medians: dict[str, float] = {}

        self.numeric_scaler = MinMaxScaler(feature_range=(-1, 1))

        self.categorical_encoder = create_one_hot_encoder(
            categories=[
                self.categorical_categories[column]
                for column in self.categorical_columns
            ]
        )

        self.target_encoder = create_one_hot_encoder(
            categories=[self.target_categories]
        )

        self.numeric_feature_names: list[str] = []
        self.categorical_feature_names: list[str] = []
        self.target_feature_names: list[str] = []
        self.feature_names: list[str] = []
        self.gan_column_names: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Fit all preprocessing components using the training data only.
        """
        self.numeric_medians = {
            column: float(train_df[column].median())
            for column in self.numeric_columns
        }

        numeric_train = self._prepare_numeric(train_df)
        categorical_train = self._prepare_categorical(train_df)
        target_train = self._prepare_target(train_df)

        self.numeric_scaler.fit(numeric_train)
        self.categorical_encoder.fit(categorical_train)
        self.target_encoder.fit(target_train)

        self.numeric_feature_names = list(self.numeric_columns)

        self.categorical_feature_names = []
        for column, categories in zip(
            self.categorical_columns,
            self.categorical_encoder.categories_,
        ):
            for category in categories:
                self.categorical_feature_names.append(f"{column}__{category}")

        self.target_feature_names = [
            f"{self.target_column}__{category}"
            for category in self.target_encoder.categories_[0]
        ]

        self.feature_names = self.numeric_feature_names + self.categorical_feature_names
        self.gan_column_names = self.feature_names + self.target_feature_names

    def transform(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        Transform a raw dataframe into arrays for model training/evaluation.

        Returns:
            features:
                Numeric + categorical features, without target.
            target_onehot:
                One-hot target.
            gan_data:
                Features + one-hot target.
            target_labels:
                Original target labels.
            processed_dataframe:
                Human-readable DataFrame of the processed vector.
        """
        numeric_data = self._prepare_numeric(df)
        categorical_data = self._prepare_categorical(df)
        target_data = self._prepare_target(df)

        numeric_scaled = self.numeric_scaler.transform(numeric_data)
        categorical_onehot = self.categorical_encoder.transform(categorical_data)
        target_onehot = self.target_encoder.transform(target_data)

        features = np.concatenate(
            [numeric_scaled, categorical_onehot],
            axis=1,
        )

        gan_data = np.concatenate(
            [features, target_onehot],
            axis=1,
        )

        processed_dataframe = pd.DataFrame(
            gan_data,
            columns=self.gan_column_names,
        )

        return {
            "features": features.astype(np.float32),
            "target_onehot": target_onehot.astype(np.float32),
            "gan_data": gan_data.astype(np.float32),
            "target_labels": df[self.target_column].to_numpy(),
            "processed_dataframe": processed_dataframe,
        }

    def fit_transform(self, train_df: pd.DataFrame) -> dict[str, Any]:
        self.fit(train_df)
        return self.transform(train_df)

    def _prepare_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        numeric_df = df[self.numeric_columns].copy()

        for column, median_value in self.numeric_medians.items():
            numeric_df[column] = numeric_df[column].fillna(median_value)

        return numeric_df

    def _prepare_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        categorical_df = df[self.categorical_columns].copy()
        categorical_df = categorical_df.fillna("Missing")
        return categorical_df

    def _prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[[self.target_column]].copy()

    def save(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, output_path)

    def save_metadata(self, output_path: str | Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        metadata = {
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "target_column": self.target_column,
            "categorical_categories": self.categorical_categories,
            "target_categories": self.target_categories,
            "numeric_medians": self.numeric_medians,
            "numeric_feature_names": self.numeric_feature_names,
            "categorical_feature_names": self.categorical_feature_names,
            "target_feature_names": self.target_feature_names,
            "feature_names": self.feature_names,
            "gan_column_names": self.gan_column_names,
            "num_numeric_features": len(self.numeric_feature_names),
            "num_categorical_onehot_features": len(self.categorical_feature_names),
            "num_target_onehot_features": len(self.target_feature_names),
            "features_dim": len(self.feature_names),
            "gan_data_dim": len(self.gan_column_names),
        }

        output_path.write_text(
            json.dumps(metadata, indent=4),
            encoding="utf-8",
        )