import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# ==========================================
# FIXED: Dynamic Path Resolution for New Structure
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../models/train
MODELS_DIR = os.path.dirname(CURRENT_DIR)                  # .../models
PROJECT_ROOT = os.path.dirname(MODELS_DIR)                 # .../Archive (Project Root)

# Add both project root and models directory to Python's search path
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)

# Clean and correct imports
from data.data_preprocessing_baseline import prepare_baseline_pipeline, collate_fn
from classes.LyricsBaselineLSTM import LyricsBaselineLSTM
from classes.Logger import Logger

# FIXED: Using PROJECT_ROOT instead of PARENT_DIR
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
log_filename = os.path.join(RESULTS_DIR, f"baseline_train_log.txt")
sys.stdout = Logger(log_filename)

print(f"--- Output is being saved to: {log_filename} ---")

# ==========================================
# 1. Configuration and Hyperparameters
# ==========================================
BATCH_SIZE = 64
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 30
MAX_LEN = 20
VALIDATION_SPLIT = 0.2  # 20% of data for validation

# ==========================================
# 2. Setup Device (GPU if available, else CPU)
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==========================================
# 3. Load Data, Build Vocabulary
# ==========================================
# FIXED: Using PROJECT_ROOT instead of PARENT_DIR
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'lyrics_train_set.csv')

# Using the lyrics-only clean pipeline function
dataset, word_to_idx, idx_to_word = prepare_baseline_pipeline(CSV_PATH, max_len=MAX_LEN)
vocab_size = len(word_to_idx)

# ==========================================
# 4. Split Dataset and Create DataLoaders
# ==========================================
total_size = len(dataset)
val_size = int(total_size * VALIDATION_SPLIT)
train_size = total_size - val_size

# Randomly split the data into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print(f"Total samples: {total_size} | Training: {train_size} | Validation: {val_size}")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ==========================================
# 5. Initialize Model, Loss, and Optimizer
# ==========================================
model = LyricsBaselineLSTM(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 6. Training and Validation Loop
# ==========================================
print("\nStarting training...\n")
start_time = time.time()

best_val_ppl = float('inf')  # Start with infinity so any first epoch will be better
patience = 3                 # Number of epochs to wait for improvement before stopping
patience_counter = 0         # Tracks consecutive epochs without improvement
best_epoch = 0

for epoch in range(EPOCHS):
    
    # -------------------------
    # Training Phase
    # -------------------------
    model.train()
    total_train_loss = 0
    
    for batch_inputs, batch_targets in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        optimizer.zero_grad()
        
        logits, _ = model(batch_inputs)
        
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = batch_targets.view(-1)
        
        loss = criterion(logits_flat, targets_flat)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        
    avg_train_loss = total_train_loss / len(train_loader)
    
    # -------------------------
    # Validation Phase
    # -------------------------
    model.eval()
    total_val_loss = 0
    
    # Disable gradient calculation for validation to save memory and compute
    with torch.no_grad():
        for batch_inputs, batch_targets in val_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            logits, _ = model(batch_inputs)
            
            logits_flat = logits.view(-1, vocab_size)
            targets_flat = batch_targets.view(-1)
            
            loss = criterion(logits_flat, targets_flat)
            total_val_loss += loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader)
    
    # Calculate Perplexity for both train and validation
    train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()
    val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()
    
    print(f"\nEpoch [{epoch+1}/{EPOCHS}] - Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f}) | Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f})")

    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        best_epoch = epoch + 1
        patience_counter = 0  # Reset counter because we found a better model
        
        # Save the best model weights
        model_save_filename = os.path.join(CHECKPOINT_DIR, f"baseline_model_best.pth")
        torch.save(model.state_dict(), model_save_filename)
        print(f"--> [SAVED] New best Validation Perplexity ({best_val_ppl:.2f}) achieved at Epoch {best_epoch}!")
    else:
        patience_counter += 1
        print(f"--> [NO IMPROVEMENT] Validation Perplexity did not improve. Patience: {patience_counter}/{patience}")
        
    # Check if we should stop early
    if patience_counter >= patience:
        print(f"\n[EARLY STOPPING] Training stopped early at Epoch {epoch+1}. Best Model was at Epoch {best_epoch} with Val PPL: {best_val_ppl:.2f}")
        break


print("\nTraining Baseline finished successfully!")

end_time = time.time()
total_duration = end_time - start_time
minutes = int(total_duration // 60)
seconds = int(total_duration % 60)
print(f"Total Training Time: {minutes} minutes and {seconds} seconds ({total_duration:.2f} seconds total)")


print(f"\n==========================================")
print(f"BEST MODEL TRAINING METRICS:")
print(f"==========================================")
print(f"Best Model caught at Epoch: {best_epoch}")
print(f"Best Model Validation Perplexity: {best_val_ppl:.2f}")
print(f"==========================================\n")

# ==========================================
# SAVE FEATURE: Saving trained model weights
# ==========================================
model_save_filename = os.path.join(CHECKPOINT_DIR, f"baseline_model.pth")
torch.save(model.state_dict(), model_save_filename)
print(f"\n--- Model weights successfully saved to: {model_save_filename} ---")