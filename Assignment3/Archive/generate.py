import os
import sys
import torch
import torch.nn.functional as F
import numpy as np


# ==========================================
# Dynamic Path Resolution
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../Archive
MODELS_DIR = os.path.join(CURRENT_DIR, "models")

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)


# ==========================================
# Imports
# ==========================================
from models.classes.LyricsBaselineLSTM import LyricsBaselineLSTM
from models.classes.melody_variant_1 import MelodyConditionedLSTM_V1
from models.classes.melody_variant_2 import MelodyConditionedLSTM_V2

from data.data_preprocessing_melody import (
    prepare_melody_pipeline,
    load_and_clean_csv,
    extract_midi_features
)


# ==========================================
# Device Setup
# CUDA = NVIDIA GPU
# MPS = Apple Silicon GPU on Mac
# CPU = fallback
# ==========================================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ==========================================
# Safe Checkpoint Loading
# ==========================================
def load_checkpoint(model, checkpoint_path, device, model_name):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"{model_name} checkpoint was not found at:\n{checkpoint_path}"
        )

    try:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        state_dict = torch.load(
            checkpoint_path,
            map_location=device
        )

    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded {model_name} checkpoint from: {checkpoint_path}")


# ==========================================
# Sampling Strategies
# ==========================================
def sample_proportional(logits):
    """
    Basic proportional sampling.
    The next word is sampled according to the probability distribution.
    """
    probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
    return np.random.choice(len(probs), p=probs)


def sample_temperature(logits, temperature=0.7):
    """
    Temperature-scaled sampling.

    Lower temperature: more conservative and repetitive.
    Higher temperature: more random and diverse.
    """
    if temperature <= 0.0:
        return torch.argmax(logits).item()

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1).detach().cpu().numpy()

    return np.random.choice(len(probs), p=probs)


def sample_top_k(logits, k=10, temperature=1.0):
    """
    Top-k sampling.
    Only the k most probable words are kept before sampling.
    """
    if temperature > 0.0:
        logits = logits / temperature

    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=-1).detach().cpu().numpy()

    chosen_index = np.random.choice(k, p=probs)
    return indices[chosen_index].item()


# ==========================================
# Utility Functions
# ==========================================
def ensure_melody_tensor(melody_features):
    """
    Ensures that melody features are a float tensor with shape (1, melody_dim).
    """
    if melody_features is None:
        melody_features = torch.zeros(12, dtype=torch.float32)

    if isinstance(melody_features, np.ndarray):
        melody_features = torch.tensor(melody_features, dtype=torch.float32)

    if not isinstance(melody_features, torch.Tensor):
        melody_features = torch.tensor(melody_features, dtype=torch.float32)

    melody_features = melody_features.float()

    if melody_features.dim() == 1:
        melody_features = melody_features.unsqueeze(0)

    return melody_features


def format_lyrics(generated_words, words_per_line=7):
    """
    Formats generated words into lyric-like lines.

    The special generated token '\n' represents a learned line break.
    If the model does not generate line breaks often enough, we also enforce
    a maximum number of words per line.
    """
    lines = []
    current_line = []

    for word in generated_words:
        if word == "\n":
            if current_line:
                lines.append(" ".join(current_line))
                current_line = []
            continue

        current_line.append(word)

        if len(current_line) >= words_per_line:
            lines.append(" ".join(current_line))
            current_line = []

    if current_line:
        lines.append(" ".join(current_line))

    return "\n".join(lines)


def token_edit_distance(tokens_a, tokens_b):
    """
    Computes token-level Levenshtein edit distance.
    This is used as an additional quantitative measure for the melody probe.
    """
    n = len(tokens_a)
    m = len(tokens_b)

    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i

    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if tokens_a[i - 1] == tokens_b[j - 1]:
                cost = 0
            else:
                cost = 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    return dp[n][m]


def compare_outputs(text_a, text_b):
    """
    Returns quantitative comparison metrics between two generated outputs.
    """
    tokens_a = text_a.lower().split()
    tokens_b = text_b.lower().split()

    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if len(set_a.union(set_b)) == 0:
        jaccard_similarity = 1.0
    else:
        jaccard_similarity = len(set_a.intersection(set_b)) / len(set_a.union(set_b))

    lexical_shift = (1.0 - jaccard_similarity) * 100.0

    edit_distance = token_edit_distance(tokens_a, tokens_b)
    max_len = max(len(tokens_a), len(tokens_b), 1)
    normalized_edit_distance = edit_distance / max_len

    return {
        "jaccard_similarity": jaccard_similarity,
        "lexical_shift": lexical_shift,
        "edit_distance": edit_distance,
        "normalized_edit_distance": normalized_edit_distance
    }


def find_matching_midi(song_title, midi_files, midi_dir):
    """
    Tries to find the MIDI file that matches the song title.
    """
    clean_title = str(song_title).lower().strip()
    clean_title_underscore = clean_title.replace(" ", "_")

    for midi_file in midi_files:
        midi_name = midi_file.lower()
        midi_name_no_ext = midi_name.replace(".mid", "").replace(".midi", "")
        midi_name_spaces = midi_name_no_ext.replace("_", " ")

        if clean_title_underscore in midi_name:
            return os.path.join(midi_dir, midi_file)

        if clean_title in midi_name_spaces:
            return os.path.join(midi_dir, midi_file)

        if midi_name_spaces in clean_title:
            return os.path.join(midi_dir, midi_file)

    return None


# ==========================================
# Generation Function
# ==========================================
def generate_lyrics(
    model,
    model_type,
    start_word,
    word_to_idx,
    idx_to_word,
    melody_features=None,
    strategy="temperature",
    temperature=0.7,
    k=10,
    max_words=100,
    min_words=30,
    words_per_line=7,
    device=torch.device("cpu")
):
    """
    Autoregressively generates lyrics from a start word.

    model_type:
        - "baseline"
        - "v1"
        - "v2"

    strategy:
        - "proportional"
        - "temperature"
        - "top_k"
        - "argmax"
    """
    model.eval()

    pad_idx = word_to_idx["<PAD>"]
    unk_idx = word_to_idx["<UNK>"]
    sos_idx = word_to_idx["<SOS>"]
    eos_idx = word_to_idx["<EOS>"]
    line_idx = word_to_idx["<LINE>"]

    generated_words = [start_word]

    current_idx = word_to_idx.get(start_word.lower(), unk_idx)
    input_seq = [current_idx]

    if model_type in ["v1", "v2"]:
        melody_features = ensure_melody_tensor(melody_features).to(device)

    with torch.no_grad():
        for _ in range(max_words - 1):
            x = torch.tensor(
                [input_seq],
                dtype=torch.long,
                device=device
            )

            if model_type == "baseline":
                logits, _ = model(x)

            elif model_type == "v1":
                logits = model(x, melody_features)

            elif model_type == "v2":
                logits = model(x, melody_features)

            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            next_word_logits = logits[0, -1, :]

            if strategy == "proportional":
                next_idx = sample_proportional(next_word_logits)

            elif strategy == "temperature":
                next_idx = sample_temperature(
                    next_word_logits,
                    temperature=temperature
                )

            elif strategy == "top_k":
                next_idx = sample_top_k(
                    next_word_logits,
                    k=k,
                    temperature=temperature
                )

            elif strategy == "argmax":
                next_idx = torch.argmax(next_word_logits).item()

            else:
                raise ValueError(f"Unknown decoding strategy: {strategy}")

            if next_idx == eos_idx:
                if len(generated_words) >= min_words:
                    break
                else:
                    continue

            if next_idx in [pad_idx, sos_idx]:
                continue
            
            if next_idx == line_idx:
                generated_words.append("\n")
                input_seq.append(next_idx)
                continue

            next_word = idx_to_word[next_idx]

            generated_words.append(next_word)
            input_seq.append(next_idx)

    return format_lyrics(
        generated_words,
        words_per_line=words_per_line
    )


# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    device = get_device()
    print(f"=== Initializing Generation Engine on device: {device} ===")

    # ------------------------------------------
    # Paths
    # ------------------------------------------
    TRAIN_CSV_PATH = os.path.join(CURRENT_DIR, "data", "lyrics_train_set.csv")
    TEST_CSV_PATH = os.path.join(CURRENT_DIR, "data", "lyrics_test_set.csv")
    MIDI_DIR = os.path.join(CURRENT_DIR, "data", "midi_files")
    RESULTS_DIR = os.path.join(CURRENT_DIR, "results")
    CHECKPOINT_DIR = os.path.join(CURRENT_DIR, "checkpoints")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    main_results_path = os.path.join(
        RESULTS_DIR,
        "test_generation_results.txt"
    )

    decoding_results_path = os.path.join(
        RESULTS_DIR,
        "decoding_strategies_results.txt"
    )

    melody_probe_path = os.path.join(
        RESULTS_DIR,
        "melody_influence_probe_results.txt"
    )

    # ------------------------------------------
    # Build vocabulary from TRAIN set only
    # This must match the vocabulary used during training.
    # ------------------------------------------
    _, word_to_idx, idx_to_word = prepare_melody_pipeline(
        TRAIN_CSV_PATH,
        MIDI_DIR,
        max_len=20
    )

    vocab_size = len(word_to_idx)
    print(f"Vocabulary size: {vocab_size}")

    # ------------------------------------------
    # Checkpoint paths
    # ------------------------------------------
    BASELINE_PTH = os.path.join(
        CHECKPOINT_DIR,
        "baseline_model_best.pth"
    )

    V1_PTH = os.path.join(
        CHECKPOINT_DIR,
        "melody_v1_model_best.pth"
    )

    V2_PTH = os.path.join(
        CHECKPOINT_DIR,
        "melody_v2_model_best.pth"
    )

    # ------------------------------------------
    # Load models
    # No need to reload Word2Vec here.
    # The trained embedding weights are already saved inside the checkpoints.
    # ------------------------------------------
    baseline_model = LyricsBaselineLSTM(vocab_size).to(device)
    v1_model = MelodyConditionedLSTM_V1(vocab_size).to(device)
    v2_model = MelodyConditionedLSTM_V2(vocab_size).to(device)

    load_checkpoint(
        baseline_model,
        BASELINE_PTH,
        device,
        "Best Baseline Model"
    )

    load_checkpoint(
        v1_model,
        V1_PTH,
        device,
        "Best Melody Variant 1 Model"
    )

    load_checkpoint(
        v2_model,
        V2_PTH,
        device,
        "Best Melody Variant 2 Model"
    )

    # ------------------------------------------
    # Load real test set
    # ------------------------------------------
    test_df = load_and_clean_csv(TEST_CSV_PATH)

    if "song_title" not in test_df.columns:
        raise ValueError(
            "The test CSV must contain a column named 'song_title'."
        )

    print(f"Loaded {len(test_df)} test songs from lyrics_test_set.csv")

    seed_words = ["Today", "Love", "Night"]

    midi_files = os.listdir(MIDI_DIR) if os.path.exists(MIDI_DIR) else []

    # ------------------------------------------
    # Open output files
    # ------------------------------------------
    with open(main_results_path, "w", encoding="utf-8") as f_main, \
         open(decoding_results_path, "w", encoding="utf-8") as f_decode, \
         open(melody_probe_path, "w", encoding="utf-8") as f_probe:

        # ------------------------------------------
        # File Headers
        # ------------------------------------------
        f_main.write("=" * 70 + "\n")
        f_main.write("EXPERIMENT 1: STANDARD TEST GENERATION\n")
        f_main.write("Generation strategy: Temperature sampling, temperature = 0.7\n")
        f_main.write("Models: Baseline, Melody Variant 1, Melody Variant 2\n")
        f_main.write("=" * 70 + "\n\n")

        f_decode.write("=" * 70 + "\n")
        f_decode.write("EXPERIMENT 2: DECODING STRATEGIES COMPARISON\n")
        f_decode.write("Compared strategies: proportional, temperature, top-k\n")
        f_decode.write("Model used for comparison: Melody Variant 1\n")
        f_decode.write("=" * 70 + "\n\n")

        f_probe.write("=" * 70 + "\n")
        f_probe.write("EXPERIMENT 3: MELODY INFLUENCE PROBE\n")
        f_probe.write("Goal: Test whether melody-conditioned models react to melody changes.\n")
        f_probe.write("Corruption method: replace true melody vector with a zero melody vector.\n")
        f_probe.write("Metrics: Jaccard similarity, lexical shift, token edit distance.\n")
        f_probe.write("Models tested: Melody Variant 1 and Melody Variant 2\n")
        f_probe.write("=" * 70 + "\n\n")

        # ------------------------------------------
        # Run experiments
        # ------------------------------------------
        for song_index, row in test_df.iterrows():
            song_title = str(row["song_title"])

            print(f"Processing test song: {song_title}")

            matched_midi = find_matching_midi(
                song_title=song_title,
                midi_files=midi_files,
                midi_dir=MIDI_DIR
            )

            if matched_midi is not None:
                true_melody_vector = extract_midi_features(matched_midi)
                f_main.write(f"Matched MIDI file: {os.path.basename(matched_midi)}\n")
                f_decode.write(f"Matched MIDI file: {os.path.basename(matched_midi)}\n")
                f_probe.write(f"Matched MIDI file: {os.path.basename(matched_midi)}\n")
            else:
                true_melody_vector = torch.zeros(12)
                f_main.write("Matched MIDI file: NOT FOUND, using zero melody vector\n")
                f_decode.write("Matched MIDI file: NOT FOUND, using zero melody vector\n")
                f_probe.write("Matched MIDI file: NOT FOUND, using zero melody vector\n")

            true_melody_vector = ensure_melody_tensor(true_melody_vector)
            corrupted_melody_vector = torch.zeros_like(true_melody_vector)

            f_main.write("\n" + "=" * 70 + "\n")
            f_main.write(f"TEST SONG: {song_title}\n")
            f_main.write("=" * 70 + "\n\n")

            f_decode.write("\n" + "=" * 70 + "\n")
            f_decode.write(f"TEST SONG: {song_title}\n")
            f_decode.write("=" * 70 + "\n\n")

            f_probe.write("\n" + "=" * 70 + "\n")
            f_probe.write(f"TEST SONG: {song_title}\n")
            f_probe.write("=" * 70 + "\n\n")

            for seed in seed_words:
                # ======================================================
                # Experiment 1:
                # Standard generation for all three models
                # ======================================================
                lyrics_base = generate_lyrics(
                    baseline_model,
                    "baseline",
                    seed,
                    word_to_idx,
                    idx_to_word,
                    strategy="temperature",
                    temperature=0.7,
                    device=device
                )

                lyrics_v1 = generate_lyrics(
                    v1_model,
                    "v1",
                    seed,
                    word_to_idx,
                    idx_to_word,
                    melody_features=true_melody_vector,
                    strategy="temperature",
                    temperature=0.7,
                    device=device
                )

                lyrics_v2 = generate_lyrics(
                    v2_model,
                    "v2",
                    seed,
                    word_to_idx,
                    idx_to_word,
                    melody_features=true_melody_vector,
                    strategy="temperature",
                    temperature=0.7,
                    device=device
                )

                f_main.write(f"Seed word: {seed}\n\n")
                f_main.write("[BASELINE OUTPUT]\n")
                f_main.write(lyrics_base + "\n\n")

                f_main.write("[MELODY V1 OUTPUT]\n")
                f_main.write(lyrics_v1 + "\n\n")

                f_main.write("[MELODY V2 OUTPUT]\n")
                f_main.write(lyrics_v2 + "\n\n")

                f_main.write("-" * 70 + "\n\n")

                # ======================================================
                # Experiment 2:
                # Decoding strategies
                # We run it only on the first two test songs
                # to keep the output focused and easy to analyze.
                # ======================================================
                if song_index < 2:
                    lyrics_prop = generate_lyrics(
                        v1_model,
                        "v1",
                        seed,
                        word_to_idx,
                        idx_to_word,
                        melody_features=true_melody_vector,
                        strategy="proportional",
                        device=device
                    )

                    lyrics_temp_low = generate_lyrics(
                        v1_model,
                        "v1",
                        seed,
                        word_to_idx,
                        idx_to_word,
                        melody_features=true_melody_vector,
                        strategy="temperature",
                        temperature=0.2,
                        device=device
                    )

                    lyrics_temp_mid = generate_lyrics(
                        v1_model,
                        "v1",
                        seed,
                        word_to_idx,
                        idx_to_word,
                        melody_features=true_melody_vector,
                        strategy="temperature",
                        temperature=0.7,
                        device=device
                    )

                    lyrics_temp_high = generate_lyrics(
                        v1_model,
                        "v1",
                        seed,
                        word_to_idx,
                        idx_to_word,
                        melody_features=true_melody_vector,
                        strategy="temperature",
                        temperature=1.5,
                        device=device
                    )

                    lyrics_topk = generate_lyrics(
                        v1_model,
                        "v1",
                        seed,
                        word_to_idx,
                        idx_to_word,
                        melody_features=true_melody_vector,
                        strategy="top_k",
                        k=5,
                        temperature=0.7,
                        device=device
                    )

                    f_decode.write(f"Seed word: {seed}\n\n")

                    f_decode.write("[PROPORTIONAL SAMPLING]\n")
                    f_decode.write(lyrics_prop + "\n\n")

                    f_decode.write("[LOW TEMPERATURE: 0.2]\n")
                    f_decode.write(lyrics_temp_low + "\n\n")

                    f_decode.write("[MEDIUM TEMPERATURE: 0.7]\n")
                    f_decode.write(lyrics_temp_mid + "\n\n")

                    f_decode.write("[HIGH TEMPERATURE: 1.5]\n")
                    f_decode.write(lyrics_temp_high + "\n\n")

                    f_decode.write("[TOP-K SAMPLING: k=5, temperature=0.7]\n")
                    f_decode.write(lyrics_topk + "\n\n")

                    f_decode.write("-" * 70 + "\n\n")

                # ======================================================
                # Experiment 3:
                # Melody Influence Probe for BOTH V1 and V2
                #
                # We reset random seeds before true/corrupted generation
                # so changes are more strongly linked to melody input.
                # ======================================================

                # ---------- V1 Probe ----------
                np.random.seed(42)
                torch.manual_seed(42)

                lyrics_v1_true_probe = generate_lyrics(
                    v1_model,
                    "v1",
                    seed,
                    word_to_idx,
                    idx_to_word,
                    melody_features=true_melody_vector,
                    strategy="temperature",
                    temperature=0.7,
                    device=device
                )

                np.random.seed(42)
                torch.manual_seed(42)

                lyrics_v1_corrupted = generate_lyrics(
                    v1_model,
                    "v1",
                    seed,
                    word_to_idx,
                    idx_to_word,
                    melody_features=corrupted_melody_vector,
                    strategy="temperature",
                    temperature=0.7,
                    device=device
                )

                v1_metrics = compare_outputs(
                    lyrics_v1_true_probe,
                    lyrics_v1_corrupted
                )

                # ---------- V2 Probe ----------
                np.random.seed(42)
                torch.manual_seed(42)

                lyrics_v2_true_probe = generate_lyrics(
                    v2_model,
                    "v2",
                    seed,
                    word_to_idx,
                    idx_to_word,
                    melody_features=true_melody_vector,
                    strategy="temperature",
                    temperature=0.7,
                    device=device
                )

                np.random.seed(42)
                torch.manual_seed(42)

                lyrics_v2_corrupted = generate_lyrics(
                    v2_model,
                    "v2",
                    seed,
                    word_to_idx,
                    idx_to_word,
                    melody_features=corrupted_melody_vector,
                    strategy="temperature",
                    temperature=0.7,
                    device=device
                )

                v2_metrics = compare_outputs(
                    lyrics_v2_true_probe,
                    lyrics_v2_corrupted
                )

                f_probe.write(f"Seed word: {seed}\n\n")

                f_probe.write("[V1 WITH TRUE MELODY]\n")
                f_probe.write(lyrics_v1_true_probe + "\n\n")

                f_probe.write("[V1 WITH CORRUPTED ZERO MELODY]\n")
                f_probe.write(lyrics_v1_corrupted + "\n\n")

                f_probe.write("[V1 METRICS]\n")
                f_probe.write(
                    f"Jaccard Similarity: {v1_metrics['jaccard_similarity']:.4f}\n"
                )
                f_probe.write(
                    f"Lexical Output Shift: {v1_metrics['lexical_shift']:.2f}%\n"
                )
                f_probe.write(
                    f"Token Edit Distance: {v1_metrics['edit_distance']}\n"
                )
                f_probe.write(
                    f"Normalized Edit Distance: {v1_metrics['normalized_edit_distance']:.4f}\n\n"
                )

                f_probe.write("[V2 WITH TRUE MELODY]\n")
                f_probe.write(lyrics_v2_true_probe + "\n\n")

                f_probe.write("[V2 WITH CORRUPTED ZERO MELODY]\n")
                f_probe.write(lyrics_v2_corrupted + "\n\n")

                f_probe.write("[V2 METRICS]\n")
                f_probe.write(
                    f"Jaccard Similarity: {v2_metrics['jaccard_similarity']:.4f}\n"
                )
                f_probe.write(
                    f"Lexical Output Shift: {v2_metrics['lexical_shift']:.2f}%\n"
                )
                f_probe.write(
                    f"Token Edit Distance: {v2_metrics['edit_distance']}\n"
                )
                f_probe.write(
                    f"Normalized Edit Distance: {v2_metrics['normalized_edit_distance']:.4f}\n\n"
                )

                f_probe.write("-" * 70 + "\n\n")

    print("\n[SUCCESS] All experiment result files were generated successfully.")
    print(f"1. Standard model comparison: {main_results_path}")
    print(f"2. Decoding strategies comparison: {decoding_results_path}")
    print(f"3. Melody influence probe: {melody_probe_path}")