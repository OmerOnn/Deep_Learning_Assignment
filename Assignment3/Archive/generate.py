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
                    melody_features=None, strategy='proportional', 
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
    print(f"=== Initializing Generation Test Engine on: {device} ===")
    
    # Paths configuration
    CSV_PATH = os.path.join(CURRENT_DIR, 'data', 'lyrics_train_set.csv')
    MIDI_DIR = os.path.join(CURRENT_DIR, 'data', 'midi_files')
    RESULTS_DIR = os.path.join(CURRENT_DIR, 'results')
    CHECKPOINT_DIR = os.path.join(CURRENT_DIR, 'checkpoints')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # Setup output text file in results folder
    output_text_filepath = os.path.join(RESULTS_DIR, "test_generation_results.txt")
    output_file = open(output_text_filepath, "w", encoding="utf-8")
    
    def log_and_print(message):
        print(message)
        output_file.write(message + "\n")

    # Load Vocabulary Maps
    _, word_to_idx, idx_to_word = prepare_melody_pipeline(CSV_PATH, MIDI_DIR, max_len=20)
    vocab_size = len(word_to_idx)
    
    
    BASELINE_PTH = os.path.join(CHECKPOINT_DIR, "baseline_model.pth")
    V1_PTH = os.path.join(CHECKPOINT_DIR, "melody_v1_model.pth")
    V2_PTH = os.path.join(CHECKPOINT_DIR, "melody_v2_model.pth")
    
    # Initialize and load Baseline Model
    baseline_model = LyricsBaselineLSTM(vocab_size).to(device)
    if os.path.exists(BASELINE_PTH):
        baseline_model.load_state_dict(torch.load(BASELINE_PTH, map_location=device))
        print("Loaded Baseline model weights.")
    else:
        print(f"Warning: Baseline weights file not found at {BASELINE_PTH}. Running uninitialized.")
        
    # Initialize and load Variant 1 Model
    v1_model = MelodyConditionedLSTM_V1(vocab_size).to(device)
    if os.path.exists(V1_PTH):
        v1_model.load_state_dict(torch.load(V1_PTH, map_location=device))
        print("Loaded Melody Variant 1 model weights.")
    else:
        print(f"Warning: V1 weights file not found at {V1_PTH}. Running uninitialized.")
        
    # Initialize and load Variant 2 Model
    v2_model = MelodyConditionedLSTM_V2(vocab_size).to(device)
    if os.path.exists(V2_PTH):
        v2_model.load_state_dict(torch.load(V2_PTH, map_location=device))
        print("Loaded Melody Variant 2 model weights.")
    else:
        print(f"Warning: V2 weights file not found at {V2_PTH}. Running uninitialized.")

    # Load full dataset dataframe to extract the last 5 rows (Test Set)
    df = load_and_clean_csv(CSV_PATH)
    test_df = df.tail(5) # Extracts the 5 test songs according to prompt specifications
    
    # Define test seed parameters
    seed_words = ["Today", "Love", "Night"]
    midi_files = os.listdir(MIDI_DIR) if os.path.exists(MIDI_DIR) else []
    
    log_and_print("\n" + "="*50)
    log_and_print("STARTING TEST LYRICS GENERATION FOR THE REPORT")
    log_and_print("="*50 + "\n")
    
    # Loop over each of the 5 test songs
    for idx, row in test_df.iterrows():
        song_title = str(row['song_title'])
        clean_title = song_title.lower().strip().replace(" ", "_")
        log_and_print(f"Processing Test Song: {song_title}")
        
        # Locate and extract MIDI features for this test song
        matched_midi = None
        for f in midi_files:
            if clean_title in f.lower() or f.lower().replace(".mid", "").replace("_", " ") in clean_title:
                matched_midi = os.path.join(MIDI_DIR, f)
                break
                
        melody_vector = extract_midi_features(matched_midi) if matched_midi else torch.zeros(12)
        
        # Run generation for each of the 3 required seed words
        for seed in seed_words:
            log_and_print(f"  -> Generating with seed word: '{seed}'")
            
            # 1. Generate via Baseline Model
            lyrics_base = generate_lyrics(baseline_model, 'baseline', seed, word_to_idx, idx_to_word, 
                                          strategy='temperature', temperature=0.7, device=device)
            
            # 2. Generate via Melody Variant 1
            lyrics_v1 = generate_lyrics(v1_model, 'v1', seed, word_to_idx, idx_to_word, 
                                        melody_features=melody_vector, strategy='temperature', temperature=0.7, device=device)
            
            # 3. Generate via Melody Variant 2
            lyrics_v2 = generate_lyrics(v2_model, 'v2', seed, word_to_idx, idx_to_word, 
                                        melody_features=melody_vector, strategy='temperature', temperature=0.7, device=device)
            
            # Log and print outputs
            log_and_print(f"\n    [BASELINE OUTPUT]:\n{lyrics_base}\n")
            log_and_print(f"    [MELODY V1 OUTPUT]:\n{lyrics_v1}\n")
            log_and_print(f"    [MELODY V2 OUTPUT]:\n{lyrics_v2}\n")
            log_and_print("-" * 40)
            
    output_file.close()
    print(f"\n[SUCCESS] All generated text has been saved safely to: {output_text_filepath}")