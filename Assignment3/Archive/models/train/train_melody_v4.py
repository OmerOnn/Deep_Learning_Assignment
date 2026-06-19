import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# ==========================================
# Dynamic Path Resolution for New Structure
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../models/train
MODELS_DIR = os.path.dirname(CURRENT_DIR)                  # .../models
PROJECT_ROOT = os.path.dirname(MODELS_DIR)                 # .../Archive (Project Root)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)

from data.data_preprocessing_melody import prepare_melody_pipeline, collate_fn_melody
from data.word2vec_utils import load_word2vec_model, build_embedding_matrix
from classes.melody_variant_4 import MelodyAttentionLSTM_V4
from classes.Logger import Logger

RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')
TENSORBOARD_DIR = os.path.join(RESULTS_DIR, 'tensorboard')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_DIR, exist_ok=True)

log_filename = os.path.join(RESULTS_DIR, f"melody_v4_train_log.txt")
sys.stdout = Logger(log_filename)

print(f"--- Output is being saved to: {log_filename} ---")

writer = SummaryWriter(
    log_dir=os.path.join(TENSORBOARD_DIR, "melody_v4")
)

# ==========================================
# 1. Configuration and Hyperparameters
# ==========================================
BATCH_SIZE = 64
EMBEDDING_DIM = 300
HIDDEN_DIM = 256
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 50
MAX_LEN = 20
VALIDATION_SPLIT = 0.2
DROPOUT = 0.3

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# ==========================================
# 2. Load Data Pipeline
# ==========================================
CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'lyrics_train_set.csv')
MIDI_DIR = os.path.join(PROJECT_ROOT, 'data', 'midi_files')

dataset, word_to_idx, idx_to_word = prepare_melody_pipeline(
    CSV_PATH,
    MIDI_DIR,
    max_len=MAX_LEN
)

vocab_size = len(word_to_idx)

WORD2VEC_PATH = os.path.join(
    PROJECT_ROOT,
    'data',
    'word2vec',
    'GoogleNews-vectors-negative300.bin'
)

word2vec_model = load_word2vec_model(WORD2VEC_PATH)

embedding_matrix = build_embedding_matrix(
    word_to_idx=word_to_idx,
    word2vec_model=word2vec_model,
    embedding_dim=EMBEDDING_DIM
)

total_size = len(dataset)
val_size = int(total_size * VALIDATION_SPLIT)
train_size = total_size - val_size

generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=generator
)

print(f"Total samples: {total_size} | Training: {train_size} | Validation: {val_size}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn_melody
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn_melody
)

# ==========================================
# 3. Initialize Melody V4 Model
# ==========================================
model = MelodyAttentionLSTM_V4(
    vocab_size=vocab_size,
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    melody_dim=12,
    pitch_embedding_dim=64,
    attention_dim=128,
    pretrained_embeddings=embedding_matrix,
    freeze_embeddings=False,
    dropout=DROPOUT
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ==========================================
# 4. Training and Validation Loop
# ==========================================
print("\nStarting Melody Variant 4 training...\n")
start_time = time.time()

best_val_ppl = float('inf')
best_train_loss = None
best_val_loss = None
best_train_ppl = None
best_epoch = 0

patience = 5
patience_counter = 0

model_save_filename = os.path.join(CHECKPOINT_DIR, f"melody_v4_model_best.pth")

for epoch in range(EPOCHS):

    # -------------------------
    # Training Phase
    # -------------------------
    model.train()
    total_train_loss = 0

    for batch_inputs, batch_targets, batch_melodies in train_loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        batch_melodies = batch_melodies.to(device)

        optimizer.zero_grad()

        logits = model(batch_inputs, batch_melodies)

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

    with torch.no_grad():
        for batch_inputs, batch_targets, batch_melodies in val_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_melodies = batch_melodies.to(device)

            logits = model(batch_inputs, batch_melodies)

            logits_flat = logits.view(-1, vocab_size)
            targets_flat = batch_targets.view(-1)

            loss = criterion(logits_flat, targets_flat)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    train_ppl = torch.exp(torch.tensor(avg_train_loss)).item()
    val_ppl = torch.exp(torch.tensor(avg_val_loss)).item()

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}] - "
        f"Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f}) | "
        f"Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f})"
    )

    writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
    writer.add_scalar("Loss/validation", avg_val_loss, epoch + 1)
    writer.add_scalar("Perplexity/train", train_ppl, epoch + 1)
    writer.add_scalar("Perplexity/validation", val_ppl, epoch + 1)

    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        best_train_loss = avg_train_loss
        best_val_loss = avg_val_loss
        best_train_ppl = train_ppl
        best_epoch = epoch + 1
        patience_counter = 0

        writer.add_scalar("Best/validation_loss", best_val_loss, best_epoch)
        writer.add_scalar("Best/validation_perplexity", best_val_ppl, best_epoch)

        torch.save(model.state_dict(), model_save_filename)

        print(
            f"--> [SAVED] New best model at Epoch {best_epoch} | "
            f"Train Loss: {best_train_loss:.4f} (PPL: {best_train_ppl:.2f}) | "
            f"Val Loss: {best_val_loss:.4f} (PPL: {best_val_ppl:.2f})"
        )

    else:
        patience_counter += 1
        print(
            f"--> [NO IMPROVEMENT] Validation Perplexity did not improve. "
            f"Patience: {patience_counter}/{patience}"
        )

    if patience_counter >= patience:
        print(
            f"\n[EARLY STOPPING] Training stopped early at Epoch {epoch+1}. "
            f"Best Model was at Epoch {best_epoch} with Val PPL: {best_val_ppl:.2f}"
        )
        break

print("\nTraining Melody Variant 4 finished successfully!")

end_time = time.time()
total_duration = end_time - start_time
minutes = int(total_duration // 60)
seconds = int(total_duration % 60)

print(
    f"Total Training Time: {minutes} minutes and {seconds} seconds "
    f"({total_duration:.2f} seconds total)"
)

print(f"\n==========================================")
print(f"BEST MELODY_V4 MODEL TRAINING METRICS:")
print(f"==========================================")
print(f"Best Model Epoch: {best_epoch}")
print(f"Best Model Train Loss: {best_train_loss:.4f}")
print(f"Best Model Train Perplexity: {best_train_ppl:.2f}")
print(f"Best Model Validation Loss: {best_val_loss:.4f}")
print(f"Best Model Validation Perplexity: {best_val_ppl:.2f}")
print(f"Best checkpoint saved to: {model_save_filename}")
print(f"==========================================\n")

writer.close()