import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Dynamic Path Resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, 'models')
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)

# Import model architectures and pipeline functions
from models.classes.LyricsBaselineLSTM import LyricsBaselineLSTM
from models.classes.melody_variant_1 import MelodyConditionedLSTM_V1
from models.classes.melody_variant_2 import MelodyConditionedLSTM_V2
from data.data_preprocessing_melody import prepare_melody_pipeline, load_and_clean_csv, extract_midi_features

# ==========================================
# SAMPLING STRATEGIES (Decoding)
# ==========================================

def sample_proportional(logits):
    """Basic proportional sampling based on model probabilities."""
    probs = F.softmax(logits, dim=-1).cpu().numpy()
    return np.random.choice(len(probs), p=probs)

def sample_temperature(logits, temperature=0.7):
    """Temperature-scaled sampling to control creativity/randomness."""
    if temperature <= 0.0:
        return torch.argmax(logits).item()
    
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()
    return np.random.choice(len(probs), p=probs)

def sample_top_k(logits, k=10, temperature=1.0):
    """Top-K sampling: keeps only the top K most likely words."""
    if temperature > 0.0:
        logits = logits / temperature
        
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=-1).cpu().numpy()
    chosen_index = np.random.choice(k, p=probs)
    return indices[chosen_index].item()

# ==========================================
# GENERATION ENGINE
# ==========================================

def generate_lyrics(model, model_type, start_word, word_to_idx, idx_to_word, 
                    melody_features=None, strategy='temperature', 
                    temperature=0.7, k=10, max_words=100, words_per_line=7, device='cpu'):
    
    model.eval()
    generated_words = [start_word]
    
    # Convert start word to index (fallback to <UNK> if not found)
    current_idx = word_to_idx.get(start_word.lower(), word_to_idx.get('<unk>', 1))
    input_seq = [current_idx]
    
    # Prepare melody features if applicable
    if model_type in ['v1', 'v2']:
        if melody_features is None:
            melody_features = torch.zeros(12)
        if len(melody_features.shape) == 1:
            melody_features = melody_features.unsqueeze(0) # add batch dim -> (1, 12)

    with torch.no_grad():
        for i in range(max_words - 1):
            x = torch.tensor([input_seq], dtype=torch.long).to(device)
            
            # Forward pass based on model architecture
            if model_type == 'baseline':
                logits, _ = model(x)
            elif model_type == 'v1':
                logits = model(x, melody_features.to(device))
            elif model_type == 'v2':
                logits = model(x, melody_features.to(device))
                
            # Get logits of the very last token in the generated sequence
            next_word_logits = logits[0, -1, :]
            
            # Apply chosen decoding strategy
            if strategy == 'proportional':
                next_idx = sample_proportional(next_word_logits)
            elif strategy == 'temperature':
                next_idx = sample_temperature(next_word_logits, temperature)
            elif strategy == 'top_k':
                next_idx = sample_top_k(next_word_logits, k, temperature)
            else:
                next_idx = torch.argmax(next_word_logits).item()
                
            # Stop or skip if special tags are generated
            if next_idx == word_to_idx.get('<eos>', -1):
                break
            if next_idx == word_to_idx.get('<pad>', 0) or next_idx == word_to_idx.get('<sos>', -1):
                continue
                
            next_word = idx_to_word[next_idx]
            generated_words.append(next_word)
            
            # Append token to sequence for the autoregressive loop
            input_seq.append(next_idx)
            
    # Format text into structural lyric lines (Song Structure Constraint)
    formatted_lyrics = []
    for idx, word in enumerate(generated_words):
        formatted_lyrics.append(word)
        if (idx + 1) % words_per_line == 0:
            formatted_lyrics.append("\n")
            
    return " ".join(formatted_lyrics).replace(" \n ", "\n")

# ==========================================
# MAIN EXECUTION: RUN TEST CASES & SAVE TO FILE
# ==========================================
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"=== Initializing Multi-Experiment Generation Engine on: {device} ===")
    
    # Paths configuration
    CSV_PATH = os.path.join(CURRENT_DIR, 'data', 'lyrics_train_set.csv')
    MIDI_DIR = os.path.join(CURRENT_DIR, 'data', 'midi_files')
    RESULTS_DIR = os.path.join(CURRENT_DIR, 'results')
    CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Setup the three distinct output file paths
    main_results_path = os.path.join(RESULTS_DIR, "test_generation_results.txt")
    decoding_results_path = os.path.join(RESULTS_DIR, "decoding_strategies_results.txt")
    melody_probe_path = os.path.join(RESULTS_DIR, "melody_influence_probe_results.txt")
    
    # Open all files
    f_main = open(main_results_path, "w", encoding="utf-8")
    f_decode = open(decoding_results_path, "w", encoding="utf-8")
    f_probe = open(melody_probe_path, "w", encoding="utf-8")
    
    # Load Vocabulary Maps
    _, word_to_idx, idx_to_word = prepare_melody_pipeline(CSV_PATH, MIDI_DIR, max_len=20)
    vocab_size = len(word_to_idx)
    
    # Checkpoints Configuration
    BASELINE_PTH = os.path.join(CHECKPOINT_DIR, "baseline_model_best.pth")
    V1_PTH = os.path.join(CHECKPOINT_DIR, "melody_v1_model_best.pth")
    V2_PTH = os.path.join(CHECKPOINT_DIR, "melody_v2_model_best.pth")
    
    # Load Models
    baseline_model = LyricsBaselineLSTM(vocab_size).to(device)
    if os.path.exists(BASELINE_PTH):
        baseline_model.load_state_dict(torch.load(BASELINE_PTH, map_location=device))
        print("Loaded Best Baseline model weights.")
        
    v1_model = MelodyConditionedLSTM_V1(vocab_size).to(device)
    if os.path.exists(V1_PTH):
        v1_model.load_state_dict(torch.load(V1_PTH, map_location=device))
        print("Loaded Best Melody Variant 1 model weights.")
        
    v2_model = MelodyConditionedLSTM_V2(vocab_size).to(device)
    if os.path.exists(V2_PTH):
        v2_model.load_state_dict(torch.load(V2_PTH, map_location=device))
        print("Loaded Best Melody Variant 2 model weights.")

    # Extract 5 test songs
    df = load_and_clean_csv(CSV_PATH)
    test_df = df.tail(5)
    
    seed_words = ["Today", "Love", "Night"]
    midi_files = os.listdir(MIDI_DIR) if os.path.exists(MIDI_DIR) else []
    
    print("\n>>> Running Experiment 1: Standard Qualitative Evaluation (Saved to test_generation_results.txt)...")
    f_main.write("="*60 + "\nEXPERIMENT 1: STANDARD QUALITATIVE EVALUATION (Temperature = 0.7)\n" + "="*60 + "\n\n")
    
    print(">>> Running Experiment 2: Decoding Strategies Analysis (Saved to decoding_strategies_results.txt)...")
    f_decode.write("="*60 + "\nEXPERIMENT 2: DECODING STRATEGIES ANALYSIS\n" + "="*60 + "\n\n")
    
    print(">>> Running Experiment 3: Melody Influence Probe (Saved to melody_influence_probe_results.txt)...")
    f_probe.write("="*60 + "\nEXPERIMENT 3: CONTROVERSIAL MELODY INFLUENCE PROBE\n" + "="*60 + "\n\n")

    # Loop over test songs
    for idx, row in test_df.iterrows():
        song_title = str(row['song_title'])
        clean_title = song_title.lower().strip().replace(" ", "_")
        
        f_main.write(f"Processing Test Song: {song_title}\n")
        f_decode.write(f"Processing Test Song: {song_title}\n")
        f_probe.write(f"Processing Test Song: {song_title}\n")
        
        # Extract true MIDI features
        matched_midi = None
        for f in midi_files:
            if clean_title in f.lower() or f.lower().replace(".mid", "").replace("_", " ") in clean_title:
                matched_midi = os.path.join(MIDI_DIR, f)
                break
                
        true_melody_vector = extract_midi_features(matched_midi) if matched_midi else torch.zeros(12)
        
        # Corrupted Melody Vector (Experiment 3 Probe): Vector of pure zeros (mismatched/silence profile)
        corrupted_melody_vector = torch.zeros(12)

        for seed in seed_words:
            # ----------------------------------------------------
            # EXPERIMENT 1: Standard Generation (Temperature = 0.7)
            # ----------------------------------------------------
            lyrics_base = generate_lyrics(baseline_model, 'baseline', seed, word_to_idx, idx_to_word, strategy='temperature', temperature=0.7, device=device)
            lyrics_v1 = generate_lyrics(v1_model, 'v1', seed, word_to_idx, idx_to_word, melody_features=true_melody_vector, strategy='temperature', temperature=0.7, device=device)
            lyrics_v2 = generate_lyrics(v2_model, 'v2', seed, word_to_idx, idx_to_word, melody_features=true_melody_vector, strategy='temperature', temperature=0.7, device=device)
            
            f_main.write(f"  -> Seed word: '{seed}'\n")
            f_main.write(f"    [BASELINE OUTPUT]:\n{lyrics_base}\n\n")
            f_main.write(f"    [MELODY V1 OUTPUT]:\n{lyrics_v1}\n\n")
            f_main.write(f"    [MELODY V2 OUTPUT]:\n{lyrics_v2}\n\n")
            f_main.write("-" * 50 + "\n")
            
            # ----------------------------------------------------
            # EXPERIMENT 2: Decoding Strategies Evaluation (Tested on Melody V1)
            # ----------------------------------------------------
            lyrics_prop = generate_lyrics(v1_model, 'v1', seed, word_to_idx, idx_to_word, melody_features=true_melody_vector, strategy='proportional', device=device)
            lyrics_temp_low = generate_lyrics(v1_model, 'v1', seed, word_to_idx, idx_to_word, melody_features=true_melody_vector, strategy='temperature', temperature=0.2, device=device)
            lyrics_temp_high = generate_lyrics(v1_model, 'v1', seed, word_to_idx, idx_to_word, melody_features=true_melody_vector, strategy='temperature', temperature=1.5, device=device)
            lyrics_topk = generate_lyrics(v1_model, 'v1', seed, word_to_idx, idx_to_word, melody_features=true_melody_vector, strategy='top_k', k=5, temperature=0.7, device=device)
            
            f_decode.write(f"  -> Seed word: '{seed}'\n")
            f_decode.write(f"    [STRATEGY: PROPORTIONAL]:\n{lyrics_prop}\n\n")
            f_decode.write(f"    [STRATEGY: LOW TEMPERATURE 0.2 (Deterministic/Repetitive)]:\n{lyrics_temp_low}\n\n")
            f_decode.write(f"    [STRATEGY: HIGH TEMPERATURE 1.5 (Creative/Chaotic)]:\n{lyrics_temp_high}\n\n")
            f_decode.write(f"    [STRATEGY: TOP-K (K=5)]:\n{lyrics_topk}\n\n")
            f_decode.write("-" * 50 + "\n")
            
            # ----------------------------------------------------
            # EXPERIMENT 3: Melody Influence Probe
            # ----------------------------------------------------
            # We compare V2 with true melody vs V2 with completely silent/corrupted melody
            lyrics_v2_corrupted = generate_lyrics(v2_model, 'v2', seed, word_to_idx, idx_to_word, melody_features=corrupted_melody_vector, strategy='temperature', temperature=0.7, device=device)
            
            # Quantitative Metric: Jaccard Word Similarity (Intersection over Union) to measure alignment change
            words_true = set(lyrics_v2.lower().split())
            words_corr = set(lyrics_v2_corrupted.lower().split())
            
            if len(words_true.union(words_corr)) > 0:
                jaccard_sim = len(words_true.intersection(words_corr)) / len(words_true.union(words_corr))
            else:
                jaccard_sim = 1.0
                
            lexical_shift = (1.0 - jaccard_sim) * 100 # Percentage of lexical vocabulary that changed
            
            f_probe.write(f"  -> Seed word: '{seed}'\n")
            f_probe.write(f"    [V2 WITH TRUE MELODY]:\n{lyrics_v2}\n\n")
            f_probe.write(f"    [V2 WITH CORRUPTED (ZERO) MELODY]:\n{lyrics_v2_corrupted}\n\n")
            f_probe.write(f"    >> QUANTITATIVE PROBE METRIC:\n")
            f_probe.write(f"       Vocabulary Jaccard Similarity: {jaccard_sim:.4f}\n")
            f_probe.write(f"       Lexical Output Shift: {lexical_shift:.2f}% of words changed due to melody corruption.\n")
            f_probe.write("-" * 50 + "\n")

    # Close all files cleanly
    f_main.close()
    f_decode.close()
    f_probe.close()
    
    print(f"\n[SUCCESS] All 3 experiment result files generated successfully inside: {RESULTS_DIR}")
    print("1. Standard Model Comparison -> test_generation_results.txt")
    print("2. Decoding Strategies Analysis -> decoding_strategies_results.txt")
    print("3. Melody Corruption Probe -> melody_influence_probe_results.txt")