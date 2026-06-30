import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running the script from the project root without installing src as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (
    COLLAPSED_D_UPDATES_PER_G_UPDATE,
    COLLAPSED_DISCRIMINATOR_HIDDEN_DIMS,
    COLLAPSED_GENERATOR_HIDDEN_DIMS,
    COLLAPSED_LATENT_DIM,
    MITIGATION_DIVERSITY_WEIGHT,
    MODE_COLLAPSE_BATCH_SIZE,
    MODE_COLLAPSE_BETA1,
    MODE_COLLAPSE_BETA2,
    MODE_COLLAPSE_EPOCHS,
    MODE_COLLAPSE_LEARNING_RATE,
    MODE_COLLAPSE_PRINT_EVERY,
    MODE_COLLAPSE_SEED,
    MODELS_DIR,
    PREPROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from src.mode_collapse import (
    calculate_mode_collapse_indicators,
    train_collapse_experiment_gan,
)
from src.utils import ensure_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    print("=" * 80)
    print("STEP 8: Mode collapse - induce, diagnose, and mitigate")
    print("=" * 80)

    seed = MODE_COLLAPSE_SEED

    step_results_dir = ensure_dir(RESULTS_DIR / "08_mode_collapse")
    step_models_dir = ensure_dir(MODELS_DIR / "mode_collapse")

    seed_preprocessed_dir = PREPROCESSED_DATA_DIR / f"seed_{seed}"

    train_gan_path = seed_preprocessed_dir / "train_gan.npy"
    metadata_path = seed_preprocessed_dir / "preprocessing_metadata.json"

    if not train_gan_path.exists():
        raise FileNotFoundError(f"Missing train GAN file: {train_gan_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    train_gan = np.load(train_gan_path)
    metadata = load_json(metadata_path)

    numeric_dim = int(metadata["num_numeric_features"])

    print("\nExperiment setup:")
    print(f"- Seed: {seed}")
    print(f"- Train GAN shape: {train_gan.shape}")
    print(f"- Numeric dim: {numeric_dim}")
    print(f"- Epochs: {MODE_COLLAPSE_EPOCHS}")
    print(f"- Collapsed latent dim: {COLLAPSED_LATENT_DIM}")
    print(f"- Collapsed generator hidden dims: {COLLAPSED_GENERATOR_HIDDEN_DIMS}")
    print(f"- Collapsed discriminator hidden dims: {COLLAPSED_DISCRIMINATOR_HIDDEN_DIMS}")
    print(f"- D updates per G update: {COLLAPSED_D_UPDATES_PER_G_UPDATE}")
    print(f"- Mitigation diversity weight: {MITIGATION_DIVERSITY_WEIGHT}")

    report_lines = []
    report_lines.append("STEP 8: Mode collapse - induce, diagnose, and mitigate")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Collapse indicators:")
    report_lines.append(
        "1. Categorical combination coverage = number of unique synthetic categorical-value "
        "combinations divided by the number of unique real categorical-value combinations. "
        "Lower coverage indicates that the generator covers fewer discrete modes."
    )
    report_lines.append(
        "2. Mean numeric variance ratio = average over numeric features of "
        "synthetic variance divided by real variance. Lower values indicate reduced numeric diversity."
    )
    report_lines.append("")
    report_lines.append("Collapse induction:")
    report_lines.append(
        "We deliberately reduce the generator latent dimension and capacity, and train the discriminator "
        "multiple times for each generator update. This is expected to make the discriminator too strong "
        "and push the generator toward a narrow set of outputs."
    )
    report_lines.append("")
    report_lines.append("Mitigation prediction:")
    report_lines.append(
        "We apply an explicit diversity penalty to the generator objective. If the mitigation works, "
        "categorical combination coverage, mean numeric variance ratio, and the generated batch diversity "
        "score should increase compared to the collapsed setup."
    )
    report_lines.append("")

    summary_rows = []

    experiments = [
        {
            "name": "collapsed",
            "diversity_weight": 0.0,
        },
        {
            "name": "mitigated_diversity_penalty",
            "diversity_weight": MITIGATION_DIVERSITY_WEIGHT,
        },
    ]

    for experiment in experiments:
        experiment_name = experiment["name"]
        diversity_weight = experiment["diversity_weight"]

        print("\n" + "=" * 80)
        print(f"Running experiment: {experiment_name}")
        print("=" * 80)

        experiment_results_dir = ensure_dir(step_results_dir / experiment_name)
        experiment_models_dir = ensure_dir(step_models_dir / experiment_name / f"seed_{seed}")

        result = train_collapse_experiment_gan(
            train_array=train_gan,
            numeric_dim=numeric_dim,
            latent_dim=COLLAPSED_LATENT_DIM,
            generator_hidden_dims=COLLAPSED_GENERATOR_HIDDEN_DIMS,
            discriminator_hidden_dims=COLLAPSED_DISCRIMINATOR_HIDDEN_DIMS,
            discriminator_updates_per_generator_update=COLLAPSED_D_UPDATES_PER_G_UPDATE,
            batch_size=MODE_COLLAPSE_BATCH_SIZE,
            epochs=MODE_COLLAPSE_EPOCHS,
            learning_rate=MODE_COLLAPSE_LEARNING_RATE,
            beta1=MODE_COLLAPSE_BETA1,
            beta2=MODE_COLLAPSE_BETA2,
            seed=seed,
            print_every=MODE_COLLAPSE_PRINT_EVERY,
            output_dir=experiment_models_dir,
            diversity_weight=diversity_weight,
        )

        synthetic_raw = result["synthetic_raw"]
        history_df = result["history"]

        np.save(experiment_results_dir / "synthetic_raw.npy", synthetic_raw)
        history_df.to_csv(experiment_results_dir / "training_history.csv", index=False)

        indicators = calculate_mode_collapse_indicators(
            real_gan=train_gan,
            synthetic_gan_raw=synthetic_raw,
            metadata=metadata,
        )

        pd.DataFrame([indicators]).to_csv(
            experiment_results_dir / "collapse_indicators.csv",
            index=False,
        )

        final_row = history_df.iloc[-1].to_dict()

        print("\nCollapse indicators:")
        for key, value in indicators.items():
            print(f"- {key}: {value}")

        print("\nFinal training metrics:")
        print(f"- Final discriminator loss: {final_row['discriminator_loss']:.6f}")
        print(f"- Final generator adversarial loss: {final_row['generator_adversarial_loss']:.6f}")
        print(f"- Final generator total loss: {final_row['generator_loss']:.6f}")
        print(f"- Final batch diversity score: {final_row['batch_diversity_score']:.6f}")
        print(f"- Final D(real): {final_row['d_real_score']:.6f}")
        print(f"- Final D(fake): {final_row['d_fake_score']:.6f}")

        summary_row = {
            "experiment": experiment_name,
            "seed": seed,
            "epochs": MODE_COLLAPSE_EPOCHS,
            "latent_dim": COLLAPSED_LATENT_DIM,
            "generator_hidden_dims": str(COLLAPSED_GENERATOR_HIDDEN_DIMS),
            "discriminator_hidden_dims": str(COLLAPSED_DISCRIMINATOR_HIDDEN_DIMS),
            "d_updates_per_g_update": COLLAPSED_D_UPDATES_PER_G_UPDATE,
            "diversity_weight": diversity_weight,
            "final_discriminator_loss": final_row["discriminator_loss"],
            "final_generator_adversarial_loss": final_row["generator_adversarial_loss"],
            "final_generator_total_loss": final_row["generator_loss"],
            "final_batch_diversity_score": final_row["batch_diversity_score"],
            "final_d_real_score": final_row["d_real_score"],
            "final_d_fake_score": final_row["d_fake_score"],
            **indicators,
        }

        summary_rows.append(summary_row)

        report_lines.append(f"Experiment: {experiment_name}")
        report_lines.append("-" * 80)
        report_lines.append(f"Diversity weight: {diversity_weight}")
        report_lines.append("")
        report_lines.append("Collapse indicators:")
        for key, value in indicators.items():
            report_lines.append(f"- {key}: {value}")
        report_lines.append("")
        report_lines.append("Final training metrics:")
        report_lines.append(f"- Final discriminator loss: {final_row['discriminator_loss']:.6f}")
        report_lines.append(f"- Final generator adversarial loss: {final_row['generator_adversarial_loss']:.6f}")
        report_lines.append(f"- Final generator total loss: {final_row['generator_loss']:.6f}")
        report_lines.append(f"- Final batch diversity score: {final_row['batch_diversity_score']:.6f}")
        report_lines.append(f"- Final D(real): {final_row['d_real_score']:.6f}")
        report_lines.append(f"- Final D(fake): {final_row['d_fake_score']:.6f}")
        report_lines.append("")
        report_lines.append(f"Saved result files to: {experiment_results_dir}")
        report_lines.append(f"Saved model files to: {experiment_models_dir}")
        report_lines.append("")

    summary_df = pd.DataFrame(summary_rows)

    summary_path = step_results_dir / "mode_collapse_summary.csv"
    report_path = step_results_dir / "mode_collapse_report.txt"

    summary_df.to_csv(summary_path, index=False)

    report_lines.append("=" * 80)
    report_lines.append("Summary")
    report_lines.append("=" * 80)
    report_lines.append(summary_df.to_string(index=False))
    report_lines.append("")

    if len(summary_df) == 2:
        collapsed = summary_df[summary_df["experiment"] == "collapsed"].iloc[0]
        mitigated = summary_df[summary_df["experiment"] == "mitigated_diversity_penalty"].iloc[0]

        report_lines.append("Mitigation analysis:")
        report_lines.append("-" * 80)
        report_lines.append(
            f"Categorical coverage changed from "
            f"{collapsed['categorical_combination_coverage']:.6f} to "
            f"{mitigated['categorical_combination_coverage']:.6f}."
        )
        report_lines.append(
            f"Mean numeric variance ratio changed from "
            f"{collapsed['mean_numeric_variance_ratio']:.6f} to "
            f"{mitigated['mean_numeric_variance_ratio']:.6f}."
        )
        report_lines.append(
            f"Batch diversity score changed from "
            f"{collapsed['final_batch_diversity_score']:.6f} to "
            f"{mitigated['final_batch_diversity_score']:.6f}."
        )

    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Saved step 8 results")
    print("=" * 80)
    print(f"- {summary_path}")
    print(f"- {report_path}")
    print("\nSTEP 8 completed successfully.")


if __name__ == "__main__":
    main()