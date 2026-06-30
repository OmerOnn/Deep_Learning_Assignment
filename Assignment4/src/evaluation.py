from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def onehot_to_label_indices(onehot: np.ndarray) -> np.ndarray:
    """
    Convert one-hot or soft one-hot vectors to integer labels using argmax.
    """
    return np.argmax(onehot, axis=1)


def split_gan_data(
    gan_data: np.ndarray,
    feature_dim: int,
    condition_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a full GAN-style vector into features and target one-hot part.
    """
    features = gan_data[:, :feature_dim]
    targets = gan_data[:, feature_dim:feature_dim + condition_dim]
    return features, targets

def harden_onehot_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Convert soft one-hot-like rows into hard one-hot rows using argmax.
    """
    hard = np.zeros_like(matrix)
    indices = np.argmax(matrix, axis=1)
    hard[np.arange(matrix.shape[0]), indices] = 1.0
    return hard


def harden_categorical_feature_blocks(
    features: np.ndarray,
    categorical_columns: list[str],
    categorical_categories: dict[str, list[str]],
    numeric_dim: int,
) -> np.ndarray:
    """
    Convert synthetic soft categorical blocks into hard one-hot blocks.

    Numeric features are left unchanged.
    """
    hardened = features.copy()
    start = numeric_dim

    for column in categorical_columns:
        categories = categorical_categories[column]
        end = start + len(categories)

        block = hardened[:, start:end]
        hardened[:, start:end] = harden_onehot_matrix(block)

        start = end

    return hardened


def harden_gan_data(
    gan_data: np.ndarray,
    metadata: dict,
) -> np.ndarray:
    """
    Convert a GAN-style vector into a comparable tabular representation.

    The input layout is:
        [numeric features, categorical soft/hard one-hot blocks, target soft/hard one-hot]

    The output keeps numeric features unchanged and hard-decodes all categorical
    and target one-hot blocks.
    """
    feature_dim = int(metadata["features_dim"])
    condition_dim = int(metadata["num_target_onehot_features"])
    numeric_dim = int(metadata["num_numeric_features"])

    features, target_onehot = split_gan_data(
        gan_data=gan_data,
        feature_dim=feature_dim,
        condition_dim=condition_dim,
    )

    hardened_features = harden_categorical_feature_blocks(
        features=features,
        categorical_columns=metadata["categorical_columns"],
        categorical_categories=metadata["categorical_categories"],
        numeric_dim=numeric_dim,
    )

    hardened_target = harden_onehot_matrix(target_onehot)

    return np.concatenate(
        [hardened_features, hardened_target],
        axis=1,
    ).astype(np.float32)

def run_detection_auc(
    real_data: np.ndarray,
    synthetic_data: np.ndarray,
    num_folds: int,
    random_seed: int,
    n_estimators: int,
    max_depth: int | None,
    n_jobs: int,
) -> tuple[float, float, pd.DataFrame]:
    """
    Detection metric.

    A Random Forest is trained to distinguish real samples from synthetic samples.

    Low AUC is good here.
    AUC close to 0.5 means the detector cannot reliably distinguish real from synthetic.
    """
    if real_data.shape != synthetic_data.shape:
        raise ValueError(
            f"Real and synthetic shapes must match. "
            f"Got real={real_data.shape}, synthetic={synthetic_data.shape}"
        )

    kfold = KFold(
        n_splits=num_folds,
        shuffle=True,
        random_state=random_seed,
    )

    fold_rows = []

    for fold_idx, (train_indices, test_indices) in enumerate(kfold.split(real_data), start=1):
        real_train = real_data[train_indices]
        real_test = real_data[test_indices]

        synthetic_train = synthetic_data[train_indices]
        synthetic_test = synthetic_data[test_indices]

        x_train = np.vstack([real_train, synthetic_train])
        y_train = np.concatenate([
            np.ones(real_train.shape[0]),
            np.zeros(synthetic_train.shape[0]),
        ])

        x_test = np.vstack([real_test, synthetic_test])
        y_test = np.concatenate([
            np.ones(real_test.shape[0]),
            np.zeros(synthetic_test.shape[0]),
        ])

        detector = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed + fold_idx,
            n_jobs=n_jobs,
        )

        detector.fit(x_train, y_train)

        y_score = detector.predict_proba(x_test)[:, 1]
        auc = roc_auc_score(y_test, y_score)

        fold_rows.append(
            {
                "fold": fold_idx,
                "auc": auc,
                "train_real_rows": real_train.shape[0],
                "train_synthetic_rows": synthetic_train.shape[0],
                "test_real_rows": real_test.shape[0],
                "test_synthetic_rows": synthetic_test.shape[0],
            }
        )

    fold_df = pd.DataFrame(fold_rows)

    return float(fold_df["auc"].mean()), float(fold_df["auc"].std()), fold_df


def run_efficacy(
    real_train_features: np.ndarray,
    real_train_target_onehot: np.ndarray,
    real_test_features: np.ndarray,
    real_test_target_onehot: np.ndarray,
    synthetic_features: np.ndarray,
    synthetic_target_onehot: np.ndarray,
    random_seed: int,
    n_estimators: int,
    max_depth: int | None,
    n_jobs: int,
) -> dict:
    """
    Efficacy metric.

    1. Train Random Forest on real train, evaluate on real test.
    2. Train Random Forest on synthetic train, evaluate on real test.
    3. Report synthetic AUC / real AUC.

    High ratio is good here, ideally close to 1.
    """
    y_real_train = onehot_to_label_indices(real_train_target_onehot)
    y_real_test = onehot_to_label_indices(real_test_target_onehot)
    y_synthetic = onehot_to_label_indices(synthetic_target_onehot)

    real_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed,
        n_jobs=n_jobs,
    )

    synthetic_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_seed + 1000,
        n_jobs=n_jobs,
    )

    real_model.fit(real_train_features, y_real_train)
    synthetic_model.fit(synthetic_features, y_synthetic)

    real_auc = roc_auc_score(
        y_real_test,
        real_model.predict_proba(real_test_features)[:, 1],
    )

    synthetic_auc = roc_auc_score(
        y_real_test,
        synthetic_model.predict_proba(real_test_features)[:, 1],
    )

    efficacy_ratio = synthetic_auc / real_auc if real_auc != 0 else np.nan

    return {
        "real_train_auc": float(real_auc),
        "synthetic_train_auc": float(synthetic_auc),
        "efficacy_ratio": float(efficacy_ratio),
    }


def get_categorical_feature_groups(
    categorical_columns: list[str],
    categorical_categories: dict[str, list[str]],
    numeric_dim: int,
) -> dict[str, tuple[int, int, list[str]]]:
    """
    Return start/end indices for each categorical feature inside the features vector.

    Feature vector layout:
        numeric features first,
        then categorical one-hot blocks.
    """
    groups = {}
    start = numeric_dim

    for column in categorical_columns:
        categories = categorical_categories[column]
        end = start + len(categories)
        groups[column] = (start, end, categories)
        start = end

    return groups


def decode_categorical_column(
    features: np.ndarray,
    start: int,
    end: int,
    categories: list[str],
) -> np.ndarray:
    """
    Decode a categorical block using argmax.
    Works both for real one-hot vectors and synthetic soft vectors.
    """
    block = features[:, start:end]
    indices = np.argmax(block, axis=1)
    categories_array = np.array(categories)
    return categories_array[indices]


def plot_numeric_histograms(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    numeric_columns: list[str],
    output_dir: Path,
    model_name: str,
    seed: int,
) -> None:
    """
    Save one histogram per numeric feature comparing real vs synthetic.
    Values are in the preprocessed scaled range [-1, 1].
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, column in enumerate(numeric_columns):
        plt.figure(figsize=(8, 5))
        plt.hist(real_features[:, idx], bins=40, alpha=0.6, label="Real", density=True)
        plt.hist(synthetic_features[:, idx], bins=40, alpha=0.6, label="Synthetic", density=True)
        plt.xlabel(f"{column} scaled value")
        plt.ylabel("Density")
        plt.title(f"{model_name} - Seed {seed} - {column}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{column}_histogram.png", dpi=200)
        plt.close()


def plot_target_distribution(
    real_target_onehot: np.ndarray,
    synthetic_target_onehot: np.ndarray,
    target_feature_names: list[str],
    output_path: Path,
    model_name: str,
    seed: int,
) -> None:
    """
    Compare real vs synthetic target label distribution.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    real_counts = real_target_onehot.sum(axis=0)
    synthetic_hard = np.zeros_like(synthetic_target_onehot)
    synthetic_hard[np.arange(synthetic_target_onehot.shape[0]), np.argmax(synthetic_target_onehot, axis=1)] = 1
    synthetic_counts = synthetic_hard.sum(axis=0)

    x = np.arange(len(target_feature_names))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, real_counts / real_counts.sum(), width, label="Real")
    plt.bar(x + width / 2, synthetic_counts / synthetic_counts.sum(), width, label="Synthetic")
    plt.xticks(x, target_feature_names, rotation=20)
    plt.ylabel("Ratio")
    plt.title(f"{model_name} - Seed {seed} - Target Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_selected_categorical_distributions(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    categorical_groups: dict[str, tuple[int, int, list[str]]],
    selected_columns: list[str],
    output_dir: Path,
    model_name: str,
    seed: int,
) -> None:
    """
    Save bar plots for selected categorical features.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for column in selected_columns:
        if column not in categorical_groups:
            continue

        start, end, categories = categorical_groups[column]

        real_decoded = decode_categorical_column(real_features, start, end, categories)
        synthetic_decoded = decode_categorical_column(synthetic_features, start, end, categories)

        real_counts = pd.Series(real_decoded).value_counts(normalize=True)
        synthetic_counts = pd.Series(synthetic_decoded).value_counts(normalize=True)

        plot_df = pd.DataFrame(
            {
                "real": real_counts,
                "synthetic": synthetic_counts,
            }
        ).fillna(0.0)

        plot_df = plot_df.loc[plot_df.sum(axis=1).sort_values(ascending=False).index]

        plt.figure(figsize=(10, 6))
        x = np.arange(len(plot_df.index))
        width = 0.35

        plt.bar(x - width / 2, plot_df["real"].values, width, label="Real")
        plt.bar(x + width / 2, plot_df["synthetic"].values, width, label="Synthetic")
        plt.xticks(x, plot_df.index, rotation=45, ha="right")
        plt.ylabel("Ratio")
        plt.title(f"{model_name} - Seed {seed} - {column}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{column}_distribution.png", dpi=200)
        plt.close()


def plot_numeric_correlation_matrices(
    real_features: np.ndarray,
    synthetic_features: np.ndarray,
    numeric_columns: list[str],
    output_dir: Path,
    model_name: str,
    seed: int,
) -> None:
    """
    Save numeric-only correlation matrices for real and synthetic data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    real_numeric_df = pd.DataFrame(
        real_features[:, :len(numeric_columns)],
        columns=numeric_columns,
    )

    synthetic_numeric_df = pd.DataFrame(
        synthetic_features[:, :len(numeric_columns)],
        columns=numeric_columns,
    )

    for data_name, df in [("real", real_numeric_df), ("synthetic", synthetic_numeric_df)]:
        corr = df.corr()

        plt.figure(figsize=(8, 6))
        plt.imshow(corr, vmin=-1, vmax=1)
        plt.colorbar(label="Correlation")
        plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=45, ha="right")
        plt.yticks(range(len(numeric_columns)), numeric_columns)
        plt.title(f"{model_name} - Seed {seed} - {data_name.capitalize()} Numeric Correlation")
        plt.tight_layout()
        plt.savefig(output_dir / f"{data_name}_numeric_correlation.png", dpi=200)
        plt.close()

        corr.to_csv(output_dir / f"{data_name}_numeric_correlation.csv")